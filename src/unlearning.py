"""Graph unlearning methods: GDGU, GIF, IDEA."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad as autograd_grad
from torch_geometric.nn import BatchNorm
from sklearn.metrics import roc_auc_score


def compute_batch_gradient(model, loader, criterion, device):
    """Compute average gradient over a DataLoader."""
    model.train()
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


def recalibrate_batchnorm(model, loader, device):
    """Re-compute BatchNorm running statistics on modified data."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, BatchNorm)):
            module.reset_running_stats()
    model.train()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
    model.eval()


def finetune_after_gdgu(model, train_loader, val_loader, device,
                        epochs=25, lr=1e-4, pos_weights=None):
    """Recovery fine-tuning after GDGU parameter update."""
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

        if np.any(np.isnan(probs)):
            continue
        valid_cols = [c for c in range(ys.shape[1]) if len(np.unique(ys[:, c])) > 1]
        if len(valid_cols) == 0:
            metric = 0.5
        else:
            metric = np.mean([roc_auc_score(ys[:, c], probs[:, c]) for c in valid_cols])

        if metric > best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"    Fine-tune best val macro-ROC: {best_metric:.4f}")
    return model


def gdgu_feature_unlearn(model, train_loader_orig, train_loader_modified,
                         val_loader_modified, criterion, device,
                         damp=0.1, max_norm=1.0, finetune_epochs=25,
                         finetune_lr=1e-4, pos_weights=None):
    """GDGU feature unlearning (first-order gradient-difference).

    Steps:
      1. Compute delta_g = grad(modified) - grad(original)
      2. Update theta += (delta_g / lambda) * min(1, rho / ||delta_g/lambda||)
      3. Recalibrate BatchNorm on modified data
      4. Fine-tune on modified data to recover
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # Step 1: Gradient difference
    print("  Computing gradient difference...")
    grad_modified = compute_batch_gradient(model, train_loader_modified, criterion, device)
    grad_original = compute_batch_gradient(model, train_loader_orig, criterion, device)

    delta_grad = [gm - go for gm, go in zip(grad_modified, grad_original)]
    delta_norm = sum(d.norm().item()**2 for d in delta_grad) ** 0.5
    print(f"  Delta gradient norm: {delta_norm:.6f}")

    if delta_norm < 1e-10:
        print("  Delta gradient too small, skipping GDGU update")
        return model

    # Step 2: First-order update
    update = [d / damp for d in delta_grad]
    with torch.no_grad():
        update_norm = sum(u.norm().item()**2 for u in update) ** 0.5
        clip_coef = min(1.0, max_norm / (update_norm + 1e-8))
        print(f"  Update norm: {update_norm:.6f}, clip_coef: {clip_coef:.4f}")
        for p, u in zip(params, update):
            if torch.isnan(u).any():
                continue
            p.data.add_(u, alpha=clip_coef)

    # Step 3: Recalibrate BatchNorm
    recalibrate_batchnorm(model, train_loader_modified, device)

    # Step 4: Recovery fine-tuning
    print(f"  Fine-tuning {finetune_epochs} epochs on modified data...")
    model = finetune_after_gdgu(model, train_loader_modified, val_loader_modified,
                                device, epochs=finetune_epochs, lr=finetune_lr,
                                pos_weights=pos_weights)
    return model


# ======================================================================
#  GIF & IDEA — Influence-Function-based unlearning
# ======================================================================

def _compute_grad_for_hvp(model, loader, criterion, device, max_batches=None):
    """Compute average gradient retaining computation graph for HVP.

    Unlike compute_batch_gradient (which calls loss.backward()), this
    accumulates a single scalar loss across all batches so that
    torch.autograd.grad(..., create_graph=True) can be used.

    Args:
        max_batches: If set, only use the first N batches to limit GPU memory.
            The HVP approximation remains unbiased with sub-sampled batches.
    """
    model.train()
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


def _hvp(grad_all, model_params, h_estimate):
    """Hessian-vector product: H·h = d/dθ [∇L(θ)·h]."""
    dot = sum(torch.sum(g * h) for g, h in zip(grad_all, h_estimate))
    hv = autograd_grad(dot, model_params, retain_graph=True, allow_unused=True)
    return tuple(h if h is not None else torch.zeros_like(p)
                 for h, p in zip(hv, model_params))


def gif_unlearn(model, train_loader_orig, train_loader_modified,
                criterion, device,
                iteration=50, damp=0.01, scale=50.0, max_batches=None):
    """GIF: Graph Influence Function unlearning (Neumann-series H⁻¹Δ∇).

    Steps:
      1. Compute grad_orig and grad_modified with computation graph
      2. v = grad_orig - grad_modified
      3. Neumann iteration: h = v + (1-damp)*h - HVP/scale
      4. Update θ += h / scale
      5. Recalibrate BatchNorm
    """
    print(f"  [GIF] Computing gradients (create_graph=True, max_batches={max_batches})...")
    grad_orig, params = _compute_grad_for_hvp(
        model, train_loader_orig, criterion, device, max_batches=max_batches)
    grad_mod, _ = _compute_grad_for_hvp(
        model, train_loader_modified, criterion, device, max_batches=max_batches)

    v = tuple(go - gm for go, gm in zip(grad_orig, grad_mod))
    h_est = tuple(go - gm for go, gm in zip(grad_orig, grad_mod))

    delta_norm = sum(vi.norm().item()**2 for vi in v) ** 0.5
    print(f"  Delta gradient norm: {delta_norm:.6f}")

    print(f"  Neumann series: {iteration} iters, damp={damp}, scale={scale}")
    for _ in range(iteration):
        hv = _hvp(grad_orig, params, h_est)
        with torch.no_grad():
            h_est = tuple(
                vi + (1 - damp) * hi - hvi / scale
                for vi, hi, hvi in zip(v, h_est, hv))

    # Apply parameter update
    params_change = [h / scale for h in h_est]
    with torch.no_grad():
        update_norm = sum(pc.norm().item()**2 for pc in params_change) ** 0.5
        print(f"  GIF update norm: {update_norm:.6f}")
        for p, pc in zip(params, params_change):
            if torch.isnan(pc).any():
                continue
            p.data.add_(pc)

    # Free computation graph
    del grad_orig, grad_mod, v, h_est
    torch.cuda.empty_cache()

    recalibrate_batchnorm(model, train_loader_modified, device)
    return model


def idea_unlearn(model, train_loader_orig, train_loader_modified,
                 val_loader_modified, criterion, device,
                 iteration=50, damp=0.01, scale=50.0,
                 finetune_epochs=25, finetune_lr=1e-4,
                 pos_weights=None, max_batches=None):
    """IDEA: GIF + recovery fine-tuning.

    Steps 1-5 identical to GIF, then:
      6. Fine-tune on modified data to recover utility
    """
    model = gif_unlearn(model, train_loader_orig, train_loader_modified,
                        criterion, device, iteration, damp, scale,
                        max_batches=max_batches)

    print(f"  [IDEA] Fine-tuning {finetune_epochs} epochs on modified data...")
    model = finetune_after_gdgu(model, train_loader_modified, val_loader_modified,
                                device, epochs=finetune_epochs, lr=finetune_lr,
                                pos_weights=pos_weights)
    return model
