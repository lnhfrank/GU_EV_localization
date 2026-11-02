"""Graph unlearning methods: GDGU, GIF, IDEA."""

import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad as autograd_grad
from torch_geometric.nn import BatchNorm
from sklearn.metrics import roc_auc_score


GRAD_MODES = ('eval', 'train')


def _clock(device):
    """Wall-clock time in seconds after the queued GPU work has finished.

    Used for the per-stage times of R2-12 (T_grad, T_step, T_bn, T_ft, T_val);
    it only reads the clock, so results are unchanged.
    """
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _set_grad_mode(model, grad_mode):
    """Put the model in the mode used for a gradient pass.

    'eval' turns dropout off and makes BatchNorm use, and leave unchanged, its
    running statistics, so the gradient is that of the deployed model and is
    deterministic. 'train' reproduces the submitted code: dropout is active and
    BatchNorm normalizes with batch statistics and updates its running buffers.
    """
    if grad_mode not in GRAD_MODES:
        raise ValueError(f"grad_mode must be one of {list(GRAD_MODES)}, got {grad_mode!r}")
    if grad_mode == 'train':
        model.train()
    else:
        model.eval()


def compute_batch_gradient(model, loader, criterion, device, grad_mode='eval'):
    """Compute average gradient over a DataLoader; grad_mode as in _set_grad_mode."""
    _set_grad_mode(model, grad_mode)
    params = [p for p in model.parameters() if p.requires_grad]
    total_grad = [torch.zeros_like(p) for p in params]
    n_total = 0

    for batch in loader:
        batch = batch.to(device)
        model.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        for i, p in enumerate(params):
            if p.grad is not None:
                total_grad[i] += p.grad.clone() * batch.num_graphs
        n_total += batch.num_graphs

    return [tg / n_total for tg in total_grad]


BN_RECAL_MODES = ('cumulative', 'momentum')


def recalibrate_batchnorm(model, loader, device, mode='cumulative'):
    """Re-compute BatchNorm running statistics on modified data.

    mode='cumulative' averages the batch statistics with equal weight
    (momentum=None during the pass, restored afterwards), so no weight is left
    on the reset values. 'momentum' reproduces the submitted code: an
    exponential average with the default momentum 0.1, which leaves 0.9**n of
    the weight on the reset mean 0 and variance 1 after n batches. Dropout
    stays active in both modes, as when the running statistics were first
    accumulated during training.
    """
    if mode not in BN_RECAL_MODES:
        raise ValueError(f"mode must be one of {list(BN_RECAL_MODES)}, got {mode!r}")
    saved_momentum = {}
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, BatchNorm)):
            module.reset_running_stats()
        if mode == 'cumulative' and isinstance(module, nn.modules.batchnorm._BatchNorm):
            saved_momentum[module] = module.momentum
            module.momentum = None
    model.train()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
    for module, momentum in saved_momentum.items():
        module.momentum = momentum
    model.eval()


def finetune_after_gdgu(model, train_loader, val_loader, device,
                        epochs=25, lr=1e-4, pos_weights=None, timing=None):
    """Recovery fine-tuning after GDGU parameter update.

    timing: optional dict; T_val is set to the time spent in the validation
    passes that select the best checkpoint, part of the fine-tuning time, and
    FT_curve to the validation ROC-AUC of every station after every epoch, the
    values that the checkpoint choice already computes.
    """
    t_val = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    best_metric = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

        # Val check — macro ROC-AUC
        t_v = _clock(device)
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                all_logits.append(model(batch).cpu())
                all_y.append(batch.y.cpu())
        logits = torch.cat(all_logits).numpy()
        ys = torch.cat(all_y).numpy()
        probs = 1 / (1 + np.exp(-logits))

        curve_row = [np.nan] * ys.shape[1]
        if np.any(np.isnan(probs)):
            t_val += _clock(device) - t_v
            if timing is not None:
                timing.setdefault('FT_curve', []).append(curve_row)
            continue
        valid_cols = [c for c in range(ys.shape[1]) if len(np.unique(ys[:, c])) > 1]
        if len(valid_cols) == 0:
            metric = 0.5
            per_station = []
        else:
            per_station = [roc_auc_score(ys[:, c], probs[:, c]) for c in valid_cols]
            metric = np.mean(per_station)

        if metric > best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
        t_val += _clock(device) - t_v
        if timing is not None:
            for c, v in zip(valid_cols, per_station):
                curve_row[c] = float(v)
            timing.setdefault('FT_curve', []).append(curve_row)

    model.load_state_dict(best_state)
    if timing is not None:
        timing['T_val'] = t_val
    print(f"    Fine-tune best val macro-ROC: {best_metric:.4f}")
    return model


GDGU_SIGNS = {'descent': -1.0, 'submitted': 1.0}
GDGU_STEPS = ('clip', 'fixed', 'line_search')
GDGU_DIRECTIONS = ('delta_g', 'random')


def _random_unit(params):
    """Random direction of unit norm over params, from the torch RNG."""
    r = [torch.randn_like(p) for p in params]
    norm = sum(x.norm().item() ** 2 for x in r) ** 0.5
    return [x / norm for x in r]


def _mean_loss(model, loader, criterion, device):
    """Average loss over a DataLoader in eval mode, weighted by graphs per batch."""
    model.eval()
    total, n_total = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            total += criterion(model(batch), batch.y).item() * batch.num_graphs
            n_total += batch.num_graphs
    return total / n_total


def _line_search_length(model, params, unit, direction, grid, loader, criterion, device):
    """Return the step length in grid, or 0, that gives the lowest loss on loader.

    Each candidate t moves the parameters to theta + direction * t * unit, where
    theta is their value on entry; theta is restored before returning. The loss is
    evaluated in eval mode, so the batch-normalization statistics stay at their
    current values, as they do for the gradient difference itself.
    """
    theta0 = [p.detach().clone() for p in params]
    best_len = 0.0
    best_loss = _mean_loss(model, loader, criterion, device)
    print(f"    Line search: t=0 loss={best_loss:.6f}")
    for t in grid:
        with torch.no_grad():
            for p, p0, u in zip(params, theta0, unit):
                if torch.isnan(u).any():
                    continue
                p.copy_(p0 + (direction * t) * u)
        loss = _mean_loss(model, loader, criterion, device)
        print(f"    Line search: t={t:g} loss={loss:.6f}")
        if loss < best_loss:
            best_len, best_loss = float(t), loss
    with torch.no_grad():
        for p, p0 in zip(params, theta0):
            p.copy_(p0)
    return best_len


def gdgu_feature_unlearn(model, train_loader_orig, train_loader_modified,
                         val_loader_modified, criterion, device,
                         damp=0.1, max_norm=1.0, finetune_epochs=25,
                         finetune_lr=1e-4, pos_weights=None, sign='descent',
                         step='fixed', line_search_grid=(0.01, 0.03, 0.1, 0.3, 1.0),
                         grad_mode='eval', bn_recal='cumulative',
                         do_step=True, do_bn=True, direction='delta_g', diagnostics=None):
    """GDGU feature unlearning (first-order gradient-difference).

    Steps:
      1. Compute delta_g = grad(modified) - grad(original)
      2. Move theta along -delta_g (sign='descent'); `step` sets the length
      3. Recalibrate BatchNorm on modified data
      4. Fine-tune on modified data to recover

    Args:
        max_norm: rho, the norm cap in step='clip' and the step length in
            step='fixed'.
        sign: 'descent' moves against delta_g, a first-order step toward the
            minimizer of the loss on the modified data. 'submitted' reproduces
            the manuscript as submitted, theta += delta_g / lambda, which
            ascends that loss; it is kept only for the ablation and for
            reproducing the original results.
        step: 'clip' reproduces the submitted rule, delta_g / lambda rescaled
            to norm rho when it is longer; in the submitted runs it was always
            rescaled, so lambda had no effect. 'fixed' moves a length rho along
            the unit vector of delta_g, so lambda is not used. 'line_search'
            picks the length from line_search_grid, or no step at all, by the
            lowest loss on the modified training data.
        line_search_grid: candidate step lengths for step='line_search'.
        grad_mode: 'eval' computes both gradients of delta_g without dropout
            and without touching the BatchNorm buffers; 'train' reproduces the
            submitted code. See _set_grad_mode.
        bn_recal: 'cumulative' or 'momentum' (submitted code), see
            recalibrate_batchnorm.
        do_step, do_bn: switch the correction step and the BatchNorm
            recalibration off for the ablation arms; finetune_epochs=0 switches
            off the fine-tuning. With do_step=False no gradient is computed.
        direction: 'delta_g' steps along the gradient difference; 'random'
            replaces it with a random unit direction and keeps the step length
            that delta_g would get, the matched-norm control of R3.
        diagnostics: optional dict, filled with Dg_norm (norm of delta_g),
            Step_len (norm of the applied correction), Clip_coef (step='clip'),
            LS_t (the length chosen by step='line_search'), and the stage times
            in seconds (R2-12): T_grad for the two gradient passes, T_step for
            choosing and applying the step, T_bn, T_ft for the fine-tuning with
            its validation passes, and T_val for those passes alone. A stage
            that does not run gets 0.
    """
    if sign not in GDGU_SIGNS:
        raise ValueError(f"sign must be one of {list(GDGU_SIGNS)}, got {sign!r}")
    if step not in GDGU_STEPS:
        raise ValueError(f"step must be one of {list(GDGU_STEPS)}, got {step!r}")
    if direction not in GDGU_DIRECTIONS:
        raise ValueError(f"direction must be one of {list(GDGU_DIRECTIONS)}, got {direction!r}")
    diag = diagnostics if diagnostics is not None else {}
    for key in ('T_grad', 'T_step', 'T_bn', 'T_ft', 'T_val'):
        diag[key] = 0.0
    if do_step:
        stepped = _gdgu_step(model, train_loader_orig, train_loader_modified, criterion,
                             device, damp, max_norm, sign, step, line_search_grid,
                             grad_mode, direction, diag)
        if not stepped:
            return model
    else:
        diag['Step_len'] = 0.0

    # Step 3: Recalibrate BatchNorm
    if do_bn:
        t0 = _clock(device)
        recalibrate_batchnorm(model, train_loader_modified, device, mode=bn_recal)
        diag['T_bn'] = _clock(device) - t0

    # Step 4: Recovery fine-tuning
    if finetune_epochs > 0:
        print(f"  Fine-tuning {finetune_epochs} epochs on modified data...")
        t0 = _clock(device)
        model = finetune_after_gdgu(model, train_loader_modified, val_loader_modified,
                                    device, epochs=finetune_epochs, lr=finetune_lr,
                                    pos_weights=pos_weights, timing=diag)
        diag['T_ft'] = _clock(device) - t0
    return model


def _gdgu_step(model, train_loader_orig, train_loader_modified, criterion, device,
               damp, max_norm, sign, step, line_search_grid, grad_mode, direction_kind,
               diag):
    """Steps 1 and 2 of GDGU; returns False when delta_g vanishes and GDGU stops."""
    direction = GDGU_SIGNS[sign]
    params = [p for p in model.parameters() if p.requires_grad]

    # Step 1: Gradient difference
    print("  Computing gradient difference...")
    t0 = _clock(device)
    grad_modified = compute_batch_gradient(model, train_loader_modified, criterion, device,
                                           grad_mode=grad_mode)
    grad_original = compute_batch_gradient(model, train_loader_orig, criterion, device,
                                           grad_mode=grad_mode)
    t1 = _clock(device)
    diag['T_grad'] = t1 - t0

    delta_grad = [gm - go for gm, go in zip(grad_modified, grad_original)]
    delta_norm = sum(d.norm().item()**2 for d in delta_grad) ** 0.5
    print(f"  Delta gradient norm: {delta_norm:.6f}")
    diag['Dg_norm'] = delta_norm

    if delta_norm < 1e-10:
        print("  Delta gradient too small, skipping GDGU update")
        diag['Step_len'] = 0.0
        diag['T_step'] = _clock(device) - t1
        return False

    # Step 2: First-order update
    if step == 'clip':
        # Submitted rule, kept verbatim so that this branch reproduces it bitwise
        update = [d / damp for d in delta_grad]
        with torch.no_grad():
            update_norm = sum(u.norm().item()**2 for u in update) ** 0.5
            clip_coef = min(1.0, max_norm / (update_norm + 1e-8))
            print(f"  Update norm: {update_norm:.6f}, clip_coef: {clip_coef:.4f}, sign: {sign}")
            diag['Clip_coef'] = clip_coef
            diag['Step_len'] = clip_coef * update_norm
            if direction_kind == 'random':
                update = [u * update_norm for u in _random_unit(params)]
            for p, u in zip(params, update):
                if torch.isnan(u).any():
                    continue
                p.data.add_(u, alpha=direction * clip_coef)
    else:
        unit = [d / delta_norm for d in delta_grad]
        if step == 'fixed':
            step_len = max_norm
        else:
            step_len = _line_search_length(model, params, unit, direction, line_search_grid,
                                           train_loader_modified, criterion, device)
            diag['LS_t'] = step_len
        print(f"  Step length: {step_len:.4f} ({step}), sign: {sign}, direction: {direction_kind}")
        diag['Step_len'] = step_len
        if direction_kind == 'random':
            unit = _random_unit(params)
        with torch.no_grad():
            for p, u in zip(params, unit):
                if torch.isnan(u).any():
                    continue
                p.data.add_(u, alpha=direction * step_len)
    diag['T_step'] = _clock(device) - t1
    return True


# ======================================================================
#  GIF & IDEA — Influence-Function-based unlearning
# ======================================================================

def _compute_grad_for_hvp(model, loader, criterion, device, max_batches=None,
                          grad_mode='eval'):
    """Compute average gradient retaining computation graph for HVP.

    Unlike compute_batch_gradient (which calls loss.backward()), this
    accumulates a single scalar loss across all batches so that
    torch.autograd.grad(..., create_graph=True) can be used.

    Args:
        max_batches: If set, only use the first N batches to limit GPU memory.
            The HVP approximation remains unbiased with sub-sampled batches.
        grad_mode: 'eval' (default) or 'train' (submitted code), see _set_grad_mode.
    """
    _set_grad_mode(model, grad_mode)
    total_loss = 0.0
    n_total = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y) * batch.num_graphs
        total_loss = total_loss + loss
        n_total += batch.num_graphs
    avg_loss = total_loss / n_total
    params = [p for p in model.parameters() if p.requires_grad]
    grads = autograd_grad(avg_loss, params, create_graph=True)
    return grads, params


def _grad_values(model, loader, criterion, device, max_batches=None, grad_mode='eval'):
    """Average gradient over the first max_batches batches, without keeping a graph."""
    _set_grad_mode(model, grad_mode)
    params = [p for p in model.parameters() if p.requires_grad]
    total = [torch.zeros_like(p) for p in params]
    n_total = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device)
        loss = criterion(model(batch), batch.y) * batch.num_graphs
        for t, g in zip(total, autograd_grad(loss, params, allow_unused=True)):
            if g is not None:
                t += g
        n_total += batch.num_graphs
    return [t / n_total for t in total], params


def _hvp_per_batch(model, loader, criterion, device, vec, max_batches=None,
                   grad_mode='eval'):
    """Hessian-vector product of the average loss, one batch graph at a time.

    Equal to _hvp on the graph of all max_batches batches, since the Hessian of
    the average loss is the graph-weighted average of the batch Hessians, but
    only one batch graph is alive at any time.
    """
    _set_grad_mode(model, grad_mode)
    params = [p for p in model.parameters() if p.requires_grad]
    total = [torch.zeros_like(p) for p in params]
    n_total = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device)
        loss = criterion(model(batch), batch.y) * batch.num_graphs
        grads = autograd_grad(loss, params, create_graph=True)
        dot = sum(torch.sum(g * v) for g, v in zip(grads, vec))
        for t, h in zip(total, autograd_grad(dot, params, allow_unused=True)):
            if h is not None:
                t += h.detach()
        n_total += batch.num_graphs
    return [t / n_total for t in total]


GIF_HVP_MODES = ('per_batch', 'full')


def _hvp(grad_all, model_params, h_estimate):
    """Hessian-vector product: H·h = d/dθ [∇L(θ)·h]."""
    dot = sum(torch.sum(g * h) for g, h in zip(grad_all, h_estimate))
    hv = autograd_grad(dot, model_params, retain_graph=True, allow_unused=True)
    return tuple(h if h is not None else torch.zeros_like(p)
                 for h, p in zip(hv, model_params))


def idea_noise_std(noise_std, cert, update_norm):
    """Standard deviation of the Gaussian noise that IDEA adds to the GIF update.

    A number is used as given. 'cert' follows the bound of the IDEA code in
    OpenGU (alpha_computation): alpha1 = (m l + sqrt(m^2 + l^2 + 4 lam n t c))
    / (lam n), with m = 0 removed samples for a feature request, n training
    nodes, and t nodes within the influence hops of the requested nodes; the
    bound is zeta = alpha1 plus the norm of the update, and the noise that makes
    the update (eps, delta)-certified is sigma = zeta sqrt(2 ln(1.25/delta)) / eps
    (Theorem 3 of Dong et al.). The constants l, lam, c are assumptions that a
    GNN does not satisfy, so the value is an order-of-magnitude estimate.

    cert: dict with n_train_nodes, n_influenced, eps, delta, l, lam, c, and
    optionally m.
    """
    if noise_std != 'cert':
        return float(noise_std)
    m = cert.get('m', 0.0)
    n, t = cert['n_train_nodes'], cert['n_influenced']
    alpha1 = (m * cert['l'] + math.sqrt(m ** 2 + cert['l'] ** 2
                                        + 4 * cert['lam'] * n * t * cert['c'])) / (cert['lam'] * n)
    zeta = alpha1 + update_norm
    return zeta * math.sqrt(2 * math.log(1.25 / cert['delta'])) / cert['eps']


def gif_unlearn(model, train_loader_orig, train_loader_modified,
                criterion, device,
                iteration=50, damp=0.01, scale=50.0, max_batches=None,
                grad_mode='eval', bn_recal='cumulative', diagnostics=None,
                hvp='per_batch', noise_std=0.0, cert=None):
    """GIF: Graph Influence Function unlearning (Neumann-series H⁻¹Δ∇).

    Steps:
      1. Compute grad_orig and grad_modified with computation graph
      2. v = grad_orig - grad_modified
      3. Neumann iteration: h = v + (1-damp)*h - HVP/scale
      4. Update θ += h / scale
      5. Recalibrate BatchNorm

    grad_mode: mode of the gradient passes, as in gdgu_feature_unlearn, so
    that GDGU, GIF, and IDEA see the same gradients.
    bn_recal: mode of the BatchNorm recalibration, see recalibrate_batchnorm.
    diagnostics: optional dict, filled with Dg_norm, Step_len (update norm),
        and the stage times of gdgu_feature_unlearn: T_grad for the two
        gradients, T_step for the Neumann iterations and the update, T_bn.
    hvp: 'per_batch' computes both gradients without a second-order graph and
        every Hessian-vector product one batch at a time, so memory follows a
        single batch at the price of a forward pass per batch and iteration.
        'full' reproduces the submitted code, which keeps the second-order
        graph of all max_batches batches alive through the whole loop.
    noise_std: standard deviation of the Gaussian noise added to every updated
        parameter after the update, as IDEA does in OpenGU, or 'cert' for the
        value of idea_noise_std with `cert`. 0 is plain GIF and draws nothing
        from the RNG. The noise comes before the BatchNorm recalibration, so the
        running statistics match the final parameters. Noise_std is written to
        the diagnostics.
    """
    if hvp not in GIF_HVP_MODES:
        raise ValueError(f"hvp must be one of {list(GIF_HVP_MODES)}, got {hvp!r}")
    diag = diagnostics if diagnostics is not None else {}
    for key in ('T_grad', 'T_step', 'T_bn', 'T_ft', 'T_val'):
        diag[key] = 0.0
    t0 = _clock(device)
    if hvp == 'full':
        print(f"  [GIF] Computing gradients (create_graph=True, max_batches={max_batches})...")
        grad_orig, params = _compute_grad_for_hvp(
            model, train_loader_orig, criterion, device, max_batches=max_batches,
            grad_mode=grad_mode)
        grad_mod, _ = _compute_grad_for_hvp(
            model, train_loader_modified, criterion, device, max_batches=max_batches,
            grad_mode=grad_mode)
    else:
        print(f"  [GIF] Computing gradients (per-batch HVP, max_batches={max_batches})...")
        grad_orig, params = _grad_values(model, train_loader_orig, criterion, device,
                                         max_batches=max_batches, grad_mode=grad_mode)
        grad_mod, _ = _grad_values(model, train_loader_modified, criterion, device,
                                   max_batches=max_batches, grad_mode=grad_mode)

    t1 = _clock(device)
    diag['T_grad'] = t1 - t0
    v = tuple(go - gm for go, gm in zip(grad_orig, grad_mod))
    h_est = tuple(go - gm for go, gm in zip(grad_orig, grad_mod))

    delta_norm = sum(vi.norm().item()**2 for vi in v) ** 0.5
    print(f"  Delta gradient norm: {delta_norm:.6f}")
    diag['Dg_norm'] = delta_norm

    print(f"  Neumann series: {iteration} iters, damp={damp}, scale={scale}, hvp={hvp}")
    for _ in range(iteration):
        if hvp == 'full':
            hv = _hvp(grad_orig, params, h_est)
        else:
            hv = _hvp_per_batch(model, train_loader_orig, criterion, device, h_est,
                                max_batches=max_batches, grad_mode=grad_mode)
        with torch.no_grad():
            h_est = tuple(
                vi + (1 - damp) * hi - hvi / scale
                for vi, hi, hvi in zip(v, h_est, hv))

    # Apply parameter update
    params_change = [h / scale for h in h_est]
    with torch.no_grad():
        update_norm = sum(pc.norm().item()**2 for pc in params_change) ** 0.5
        print(f"  GIF update norm: {update_norm:.6f}")
        diag['Step_len'] = update_norm
        for p, pc in zip(params, params_change):
            if torch.isnan(pc).any():
                continue
            p.data.add_(pc)

        # IDEA as in OpenGU: Gaussian noise on the updated parameters
        sigma = idea_noise_std(noise_std, cert, update_norm)
        diag['Noise_std'] = sigma
        if sigma > 0:
            print(f"  [IDEA] Gaussian noise on the parameters, std {sigma:.4g}")
            for p in params:
                p.add_(torch.randn_like(p) * sigma)

    # Free computation graph
    del grad_orig, grad_mod, v, h_est
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t2 = _clock(device)
    diag['T_step'] = t2 - t1

    recalibrate_batchnorm(model, train_loader_modified, device, mode=bn_recal)
    diag['T_bn'] = _clock(device) - t2
    return model


def gif_arm_unlearn(model, train_loader_orig, train_loader_modified,
                    val_loader_modified, criterion, device,
                    iteration=50, damp=0.01, scale=50.0, max_batches=None,
                    grad_mode='eval', bn_recal='cumulative', hvp='per_batch',
                    noise_std=0.0, cert=None, finetune_epochs=0, finetune_lr=1e-4,
                    pos_weights=None, diagnostics=None):
    """GIF arms: the GIF update, optional IDEA noise, optional fine-tuning.

    finetune_epochs=0 and noise_std>0 is IDEA as implemented in OpenGU, whose
    update equals that of GIF plus Gaussian noise and has no fine-tuning.
    finetune_epochs=25 with noise_std=0 is GIF followed by the same fine-tuning
    as GDGU, the method labelled IDEA in the submitted manuscript (GIF+FT, the
    matched budget of R1-RA7); it runs the same calls as idea_unlearn.
    """
    diag = diagnostics if diagnostics is not None else {}
    model = gif_unlearn(model, train_loader_orig, train_loader_modified,
                        criterion, device, iteration, damp, scale,
                        max_batches=max_batches, grad_mode=grad_mode,
                        bn_recal=bn_recal, diagnostics=diag, hvp=hvp,
                        noise_std=noise_std, cert=cert)
    if finetune_epochs > 0:
        print(f"  [GIF arm] Fine-tuning {finetune_epochs} epochs on modified data...")
        t0 = _clock(device)
        model = finetune_after_gdgu(model, train_loader_modified, val_loader_modified,
                                    device, epochs=finetune_epochs, lr=finetune_lr,
                                    pos_weights=pos_weights, timing=diag)
        diag['T_ft'] = _clock(device) - t0
    return model


def idea_unlearn(model, train_loader_orig, train_loader_modified,
                 val_loader_modified, criterion, device,
                 iteration=50, damp=0.01, scale=50.0,
                 finetune_epochs=25, finetune_lr=1e-4,
                 pos_weights=None, max_batches=None, grad_mode='eval',
                 bn_recal='cumulative', diagnostics=None, hvp='per_batch'):
    """The method labelled IDEA in the submitted manuscript: GIF + recovery fine-tuning.

    Steps 1-5 identical to GIF, then:
      6. Fine-tune on modified data to recover utility

    Neither Dong et al. nor OpenGU fine-tune after the update; kept to
    reproduce the submitted runs. New runs use the GIF arms of gif_arm_unlearn:
    GIF+FT for this method and IDEA with noise for IDEA as in OpenGU.
    """
    model = gif_unlearn(model, train_loader_orig, train_loader_modified,
                        criterion, device, iteration, damp, scale,
                        max_batches=max_batches, grad_mode=grad_mode,
                        bn_recal=bn_recal, diagnostics=diagnostics, hvp=hvp)

    print(f"  [IDEA] Fine-tuning {finetune_epochs} epochs on modified data...")
    diag = diagnostics if diagnostics is not None else {}
    t0 = _clock(device)
    model = finetune_after_gdgu(model, train_loader_modified, val_loader_modified,
                                device, epochs=finetune_epochs, lr=finetune_lr,
                                pos_weights=pos_weights, timing=diag)
    diag['T_ft'] = _clock(device) - t0
    return model
