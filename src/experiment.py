"""Experiment runner: one trial (one backbone x scenario x seed)."""

import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models import MODEL_CLASSES, AuxWrapper
from src.data import (build_graphs, FORGET_SEMANTICS,
                      stratified_split_multilabel, fit_scaler)
from src.training import (kaiming_init, get_pos_weights,
                          train_model_joint, evaluate_model,
                          compute_mia_auc)
from src.unlearning import gdgu_feature_unlearn, gif_unlearn, idea_unlearn, gif_arm_unlearn
from src.noise import add_voltage_noise


def _peak_memory_mb(device):
    """Return peak GPU memory in MB since last reset, or 0 for CPU."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def _epoch_stats(epoch_logs, cap):
    """Epochs run, best epoch, and whether training ran into the epoch cap."""
    vals = [e['val_roc'] for e in epoch_logs]
    stopped_early = bool(epoch_logs) and epoch_logs[-1].get('stopped_early', False)
    return {'Epochs_run': len(vals),
            'Best_epoch': int(np.argmax(vals)) + 1 if vals else np.nan,
            'Hit_cap': len(vals) == cap and not stopped_early}


DIAG_COLUMNS = ('Dg_norm', 'Step_len', 'Clip_coef', 'LS_t',
                'T_grad', 'T_step', 'T_bn', 'T_ft', 'T_val', 'Noise_std')


def occlusion_delta_auc(model, loader_present, loader_occluded, device, forget_label_idx):
    """Drop in ROC-AUC when the charging power of the forgotten stations is zeroed at test.

    The primary measure of how much a model still relies on
    the withdrawn channel: 0 means it no longer uses it. Returns the macro
    drop, the mean drop over the forgotten stations, and the drop per station.
    """
    present = evaluate_model(model, loader_present, device)['per_roc']
    occluded = evaluate_model(model, loader_occluded, device)['per_roc']
    drops = [p - o for p, o in zip(present, occluded)]
    out = {'Occ_dAUC': float(np.mean(drops)),
           'Occ_dAUC_forget': float(np.mean([drops[i] for i in forget_label_idx]))}
    for i, d in enumerate(drops):
        out[f'Occ_dAUC_EVCS{i+1}'] = d
    return out


# Second entry of the seed sequence that draws the withdrawn records (E5)
WITHDRAW_STREAM = 7


def withdrawn_graphs(idx_train, fraction, seed):
    """Training graphs whose charging-power records of the forgotten stations are withdrawn (E5).

    A random fraction of idx_train from its own generator: one permutation per
    seed, of which the first round(fraction * n) entries are taken, so the sets
    of a seed are nested across fractions. Returns them sorted.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'withdraw_fraction must be in (0, 1], got {fraction}')
    idx = np.asarray(idx_train)
    n = max(1, int(round(fraction * len(idx))))
    perm = np.random.default_rng([seed, WITHDRAW_STREAM]).permutation(idx)
    return np.sort(perm[:n])


def _build_result(backbone, scenario, seed, method, eval_dict, mia_dict,
                   elapsed, mem_mb, forget_label_idx=None, epoch_stats=None,
                   diagnostics=None, occlusion=None):
    """Build a flat result dict from evaluation output (dynamic EVCS count).

    epoch_stats: output of _epoch_stats for the methods that train from
    scratch (Original, Retrain); the other methods get NaN.
    diagnostics: the dict filled by the unlearning method (DIAG_COLUMNS);
    missing entries get NaN.
    occlusion: output of occlusion_delta_auc, or None when it is switched off.
    """
    row = {
        'Backbone': backbone, 'Scenario': scenario, 'Seed': seed,
        'Method': method,
        'ExMatch': eval_dict['exact_match'],
        'Hamming_Acc': eval_dict['hamming_acc'],
        'Macro_ROC': eval_dict['macro_roc'],
        'Macro_F1': eval_dict['macro_f1'],
    }
    per_roc = eval_dict['per_roc']
    per_f1 = eval_dict['per_f1']
    for i, roc in enumerate(per_roc):
        row[f'ROC_EVCS{i+1}'] = roc
    for i, f1 in enumerate(per_f1):
        row[f'F1_EVCS{i+1}'] = f1

    # Forget / retain grouped metrics
    if forget_label_idx is not None:
        n_labels = len(per_f1)
        retain_idx = [i for i in range(n_labels) if i not in forget_label_idx]
        row['F1_forget'] = float(np.mean([per_f1[i] for i in forget_label_idx]))
        row['ROC_forget'] = float(np.mean([per_roc[i] for i in forget_label_idx]))
        if retain_idx:
            row['F1_retain'] = float(np.mean([per_f1[i] for i in retain_idx]))
            row['ROC_retain'] = float(np.mean([per_roc[i] for i in retain_idx]))
        else:
            row['F1_retain'] = np.nan
            row['ROC_retain'] = np.nan
    else:
        row['F1_forget'] = np.nan
        row['ROC_forget'] = np.nan
        row['F1_retain'] = np.nan
        row['ROC_retain'] = np.nan

    # MIA split (OpenGU-aligned)
    row['MIA_forget'] = mia_dict['mia_forget']
    row['MIA_retain'] = mia_dict['mia_retain']
    row['MIA_AUC'] = mia_dict['mia_overall']
    row['Time'] = elapsed
    row['Peak_Memory_MB'] = mem_mb
    stats = epoch_stats or {'Epochs_run': np.nan, 'Best_epoch': np.nan, 'Hit_cap': np.nan}
    row.update(stats)
    diag = diagnostics or {}
    for key in DIAG_COLUMNS:
        row[key] = diag.get(key, np.nan)
    if occlusion is not None:
        row.update(occlusion)
    return row


STANDARD_METHODS = ('GDGU', 'GIF', 'IDEA', 'Retrain')
# Run only when listed: WarmStart trains the original model on D' with
# the procedure and stopping rule of Retrain (R1-RA7); Retrain_b retrains from
# scratch on the same split with another seed, the reference for the
# randomness of retraining (R2-10).
EXTRA_METHODS = ('WarmStart', 'Retrain_b')
GDGU_ARM_KEYS = ('damp', 'max_norm', 'finetune_epochs', 'finetune_lr', 'sign', 'step',
                 'line_search_grid', 'grad_mode', 'bn_recal', 'do_step', 'do_bn', 'direction', 'init')
# Starting point of a GDGU arm: the original model, or a freshly initialized one for the
# from-scratch control of E5, which runs the arm's fine-tuning without the original model
ARM_INITS = ('original', 'scratch')
# Arms with base: gif run gif_arm_unlearn and may override these settings
GIF_ARM_KEYS = ('iteration', 'damp', 'scale', 'max_batches', 'grad_mode', 'bn_recal', 'hvp',
                'noise_std', 'finetune_epochs', 'finetune_lr')
CACHE_KEYS = ('bus_system', 'feature_mode', 'test_size', 'val_ratio', 'epochs', 'batch_size',
              'lr', 'weight_decay', 'patience', 'min_delta', 'scheduler_patience',
              'hidden_dim', 'n_layers', 'dropout', 'aux', 'readout', 'voltage_noise_snr_db',
              'holdout_attack_type', 'forget_semantics')
# Keys added to CACHE_KEYS after caches were built: a cache without the key was
# built under this value, so it is compared as if it had stored it.
CACHE_KEY_DEFAULTS = {'readout': 'dual_pool', 'voltage_noise_snr_db': None, 'holdout_attack_type': None,
                      'forget_semantics': 'zero_std'}
# Attack types that holdout_attack_type can hold out (E10); 0 is the normal operation
HOLDOUT_TYPES = (1, 2, 3, 4)


def build_model(backbone_name, config, data_dict, device, in_dim=None):
    """A backbone with the attack-type head, Kaiming-initialized from the current torch RNG state.

    in_dim defaults to the voltage and power features of data_dict.
    """
    ra = config.get('aux', {})
    backbone = MODEL_CLASSES[backbone_name](
        in_dim=in_dim if in_dim is not None else data_dict['n_feat'] * 2,
        hid_dim=config['hidden_dim'], out_dim=data_dict['n_evcs'], n_layers=config['n_layers'],
        dropout=config['dropout'], readout=config.get('readout', 'dual_pool'),
        station_nodes=data_dict['evcs_node_indices'])
    model = AuxWrapper(backbone, n_types=ra.get('n_attack_types', 5),
                       aux_hidden=ra.get('aux_hidden', 64)).to(device)
    kaiming_init(model)
    return model


def state_hash(model):
    """Short md5 of the parameters and buffers of a model, to trace where a sequential step started."""
    h = hashlib.md5()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:12]


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gdgu_kwargs(config, overrides=None):
    """Keyword arguments of gdgu_feature_unlearn from the config, with arm overrides."""
    kwargs = dict(
        damp=config['gdgu_damp'], max_norm=config['gdgu_max_norm'],
        finetune_epochs=config['gdgu_finetune'],
        finetune_lr=config.get('gdgu_finetune_lr', 1e-4),
        sign=config.get('gdgu_sign', 'descent'),
        step=config.get('gdgu_step', 'fixed'),
        line_search_grid=config.get('gdgu_line_search_grid', (0.01, 0.03, 0.1, 0.3, 1.0)),
        grad_mode=config.get('gdgu_grad_mode', 'eval'),
        bn_recal=config.get('gdgu_bn_recal', 'cumulative'))
    for key, value in (overrides or {}).items():
        if key not in GDGU_ARM_KEYS:
            raise ValueError(f"unknown GDGU arm setting {key!r}; allowed: {list(GDGU_ARM_KEYS)}")
        if key == 'init':  # where the arm starts, handled by run_single_trial
            continue
        kwargs[key] = value
    return kwargs


def gif_kwargs(config, overrides=None):
    """Keyword arguments of gif_arm_unlearn from the config, with arm overrides."""
    kwargs = dict(
        iteration=config.get('gif_iteration', 50), damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0), max_batches=config.get('gif_max_batches'),
        grad_mode=config.get('gif_grad_mode', 'eval'),
        bn_recal=config.get('gif_bn_recal', 'cumulative'),
        hvp=config.get('gif_hvp', 'per_batch'), noise_std=0.0, finetune_epochs=0,
        finetune_lr=config.get('gdgu_finetune_lr', 1e-4))
    for key, value in (overrides or {}).items():
        if key == 'base':
            continue
        if key not in GIF_ARM_KEYS:
            raise ValueError(f"unknown GIF arm setting {key!r}; allowed: {list(GIF_ARM_KEYS)}")
        kwargs[key] = value
    return kwargs


def influenced_node_count(edge_index, nodes, hops):
    """Nodes within `hops` of `nodes`, the nodes themselves excluded, in one graph.

    Follows find_k_hops of the IDEA code in OpenGU, which uses 2 hops for
    feature requests.
    """
    ei = np.asarray(edge_index)
    reached = np.unique(np.asarray(nodes))
    for _ in range(hops):
        reached = np.unique(np.append(reached, ei[1, np.isin(ei[0], reached)]))
    return int(len(np.setdiff1d(reached, np.asarray(nodes))))


def check_methods(methods, config):
    """Raise if a method is neither a standard method nor an arm in config['arms']."""
    arms = config.get('arms') or {}
    for name in methods:
        if name == 'Original':
            raise ValueError("Original always runs first; do not list it in methods")
        if name not in STANDARD_METHODS and name not in EXTRA_METHODS and name not in arms:
            raise ValueError(f"unknown method {name!r}; standard: {list(STANDARD_METHODS)}, "
                             f"extra: {list(EXTRA_METHODS)}, arms in the config: {list(arms)}")
        if name in arms:
            arm = arms[name] or {}
            allowed = GIF_ARM_KEYS if arm.get('base') == 'gif' else GDGU_ARM_KEYS
            unknown = [k for k in arm if k != 'base' and k not in allowed]
            if unknown:
                raise ValueError(f"arm {name!r} has unknown settings {unknown}; allowed: {list(allowed)}")
            if arm.get('init', 'original') not in ARM_INITS:
                raise ValueError(f"arm {name!r}: init must be one of {list(ARM_INITS)}, got {arm['init']!r}")


class ModelCache:
    """Disk cache of the trained Original and Retrain models.

    Original depends only on the backbone and the seed, because the split, the
    scalers, and its training data do not depend on the scenario; Retrain also
    depends on the scenario. An entry holds the state dict, the epoch logs, the
    training time, and the peak memory. The settings that shape training are
    stored with the cache, and a cache built under other settings is refused.
    """

    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        fingerprint = json.loads(json.dumps(
            {k: config.get(k, CACHE_KEY_DEFAULTS.get(k)) for k in CACHE_KEYS}))
        meta = self.root / 'fingerprint.json'
        if meta.exists():
            stored = json.loads(meta.read_text())
            stored = {k: stored.get(k, CACHE_KEY_DEFAULTS.get(k)) for k in CACHE_KEYS}
            if stored != fingerprint:
                diff = {k: (stored.get(k), fingerprint.get(k)) for k in fingerprint
                        if stored.get(k) != fingerprint.get(k)}
                raise ValueError(f"cache {self.root} was built with other settings: {diff}")
        else:
            meta.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))

    def _path(self, kind, backbone, seed, scenario=None):
        name = f"{kind}_{backbone}_{seed}" + (f"_{scenario}" if scenario else "")
        return self.root / f"{name}.pt"

    def load(self, kind, backbone, seed, scenario=None, device='cpu'):
        path = self._path(kind, backbone, seed, scenario)
        if not path.exists():
            return None
        return torch.load(path, map_location=device, weights_only=False)

    def save(self, kind, backbone, seed, scenario, model, logs, elapsed, mem_mb):
        entry = {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'logs': logs, 'time': elapsed, 'mem': mem_mb}
        torch.save(entry, self._path(kind, backbone, seed, scenario))



def run_single_trial(backbone_name, scen_key, scen_val, seed,
                              data_dict, config, device, methods=None, cache=None):
    """Run one trial: Original, then each method in `methods`.

    Uses AuxWrapper for joint loc + attack-type training. Unlearning methods
    operate on the loc head only (aux head is frozen). Reports utility
    (Macro ROC-AUC / F1), forgetting privacy (MIA-AUC), and efficiency.

    Args:
        data_dict: output of load_evcs_data + augment_with_power.
        config: flat dict with training/gdgu/gif/aux parameters. Two switches
            set the protocol: reseed_arms reseeds with the trial seed before
            every method, so arms that share fine-tuning see the same batch
            order and dropout masks; original_cache reuses one Original per
            backbone and seed across scenarios through `cache` and requires
            reseed_arms. Both False reproduce the submitted protocol, where
            only Retrain is reseeded and Original is trained in every trial;
            they are also the defaults when the keys are missing. A third
            switch, occlusion_auc, adds the occlusion delta AUC to every
            row; it defaults to False too, since its extra passes over the test
            loaders draw from the torch RNG, as every DataLoader pass does. A
            fourth, save_outputs, writes per trial the per-graph logits of
            every method, the fine-tuning validation curves, and the unlearned
            models to config['outputs_dir']; it requires reseed_arms. A fifth,
            voltage_noise_snr_db (E9), adds white Gaussian noise at that SNR in
            dB to the voltages of every graph before the scalers are fit
            (src.noise); None, the default, adds none. A sixth, withdraw_fraction
            (E5), withdraws the charging power of the forgotten stations from
            only that random fraction of the training graphs (withdrawn_graphs);
            the other training graphs and all validation and test graphs keep
            it, and Retrain, Retrain_b, and the saved outputs are filed under
            <scenario>_f<fraction>. 1.0, the default, withdraws it everywhere.
            A seventh, sequential (E4), serves the requests one after another:
            in scenario S_j every method starts from the model that the same
            method left after S_{j-1} of the same backbone and seed, read from
            the saved outputs, and the gradient-based methods take the training
            set of S_{j-1} as the data before the request; S1 starts from the
            original model as usual. Retrain and Retrain_b do not depend on the
            path and keep their cache entries under S_j; the outputs are filed
            under S_j_seq, and the rows carry the starting point. It requires
            save_outputs and a whole withdrawal; False, the default, serves
            every request from the original model. An eighth,
            holdout_attack_type (E10), holds one attack type out of training
            and validation: the training and validation graphs of the split
            lose that type, and the test set holds every graph of that type next
            to the normal test graphs, so the attack is unseen until the test;
            the rows carry Holdout_type. None, the default, holds none out. A
            ninth, forget_semantics (E10), sets how the withdrawn power is
            removed (src.data.build_graphs): 'zero_std', the default and the
            submitted operation, zeroes it after standardization; 'zero_raw'
            zeroes the raw power before standardization with the scaler of the
            training split; 'mask' zeroes it as 'zero_std' does and gives every
            graph one more input channel that marks the withdrawn stations. The
            other two settings add Forget_semantics to the rows.
        methods: names run after Original, in order: 'GDGU', 'GIF', 'IDEA',
            'Retrain', 'WarmStart', 'Retrain_b', or an arm of
            config['arms']: a GDGU run with overrides, or with base: gif a run
            of gif_arm_unlearn. A GDGU arm with init: scratch starts from
            a freshly initialized model instead of the original one, the
            from-scratch control of E5. Default STANDARD_METHODS.
        cache: ModelCache or None. Retrain is also read from and written to it.

    Returns:
        list of result dicts, dict of epoch_logs.
    """
    methods = list(methods or STANDARD_METHODS)
    check_methods(methods, config)
    reseed = config.get('reseed_arms', False)
    use_original_cache = config.get('original_cache', False) and cache is not None
    if config.get('original_cache', False) and not reseed:
        raise ValueError("original_cache requires reseed_arms, otherwise the random state "
                         "after Original depends on whether it came from the cache")
    save_outputs = config.get('save_outputs', False)
    if save_outputs and not reseed:
        raise ValueError("save_outputs requires reseed_arms, otherwise its extra passes over "
                         "the loaders change the random state of the next method")

    forget_node_indices = scen_val['forget_node_indices']
    forget_label_idx = scen_val['forget_label_indices']
    scen_label = scen_val['label']

    all_x_V = data_dict['all_x']
    # Measurement noise on the voltages (E9). Its generator is separate, so the
    # split, the initialization, and the batch order match the noise-free trial.
    v_noise = None
    if config.get('voltage_noise_snr_db') is not None:
        all_x_V, v_noise = add_voltage_noise(all_x_V, config['voltage_noise_snr_db'], seed)
    all_x_P = data_dict['all_x_P']
    all_y = data_dict['all_y']
    all_attack_type = data_dict['all_attack_type']
    edge_index = data_dict['edge_index']
    n_nodes = data_dict['n_nodes']

    ra = config.get('aux', {})
    gamma = ra.get('gamma', 0.5)
    n_attack_types = ra.get('n_attack_types', 5)
    # Semantics of the deletion (E10); 'mask' adds one input channel to every graph
    semantics = config.get('forget_semantics', 'zero_std')
    if semantics not in FORGET_SEMANTICS:
        raise ValueError(f"forget_semantics must be one of {list(FORGET_SEMANTICS)}, got {semantics!r}")
    mask_channel = semantics == 'mask'
    in_dim = data_dict['n_feat'] * 2 + int(mask_channel)  # V-concat-P (n_feat dims each), plus the mask
    lr = config['lr']

    print(f"\n{'='*70}")
    print(f"  [Trial] {backbone_name} | {scen_label} | seed={seed}")
    print(f"{'='*70}")
    if v_noise is not None:
        print(f"  Voltage noise: SNR {v_noise['V_noise_snr_db']:g} dB, "
              f"sigma {v_noise['V_noise_sigma']:.6f} p.u. (reference {v_noise['V_noise_ref']:.6f})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx_train, idx_val, idx_test = stratified_split_multilabel(
        all_y, config['test_size'], config['val_ratio'], seed)
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # Leave one attack type out (E10): the graphs of that type are neither trained
    # nor validated on; the test set holds all of them next to the normal test graphs
    holdout = config.get('holdout_attack_type')
    if holdout is not None:
        if holdout not in HOLDOUT_TYPES:
            raise ValueError(f'holdout_attack_type must be one of {list(HOLDOUT_TYPES)} or None, '
                             f'got {holdout!r}')
        t = np.asarray(all_attack_type)
        idx_train = idx_train[t[idx_train] != holdout]
        idx_val = idx_val[t[idx_val] != holdout]
        idx_test = np.sort(np.concatenate([idx_test[t[idx_test] == 0], np.where(t == holdout)[0]]))
        print(f"  Holdout: attack type {holdout}; train={len(idx_train)}, val={len(idx_val)}, "
              f"test={len(idx_test)} ({int((t[idx_test] == holdout).sum())} of type {holdout})")

    # Partial withdrawal (E5): only some records of the forgotten stations are
    # withdrawn, drawn from a separate generator so the global RNG is untouched
    frac = config.get('withdraw_fraction', 1.0)
    partial = frac != 1.0
    withdrawn = withdrawn_graphs(idx_train, frac, seed) if partial else None
    scen_tag = f'{scen_key}_f{frac:g}' if partial else scen_key
    if partial:
        print(f"  Withdrawal: fraction {frac:g}, {len(withdrawn)} of {len(idx_train)} training "
              f"graphs; validation and test keep the charging power")

    # Sequential requests (E4): the outputs of S_j are filed under S_j_seq, so a
    # sequential run never mixes with the requests served from the original model
    sequential = config.get('sequential', False)
    step = scen_val.get('n_evcs_forget', len(forget_label_idx))
    if sequential and not save_outputs:
        raise ValueError("sequential requires save_outputs, which keeps the model of every step")
    if sequential and partial:
        raise ValueError("sequential withdraws whole channels; set withdraw_fraction to 1.0")
    out_tag = f'{scen_tag}_seq' if sequential else scen_tag
    if sequential:
        print(f"  Sequential: step {step}, every method starts from "
              + ("the original model" if step == 1 else f"its model after S{step - 1}"))

    scaler_V = fit_scaler(all_x_V, idx_train)
    scaler_P = fit_scaler(all_x_P, idx_train)

    pw = get_pos_weights(all_y[idx_train], device)
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pw)

    type_counts = Counter(all_attack_type[idx_train])
    n_train = len(idx_train)
    tw = torch.tensor([n_train / (n_attack_types * max(type_counts.get(c, 1), 1))
                        for c in range(n_attack_types)], dtype=torch.float).to(device)

    bs = config['batch_size']

    def _bg(idx, forget=None):
        return build_graphs(
            all_x_V, all_x_P, all_y, all_attack_type,
            edge_index, n_nodes, idx, scaler_V, scaler_P,
            forget_P_at_node=forget, forget_semantics=semantics, mask_channel=mask_channel)

    train_full = _bg(idx_train)
    val_full = _bg(idx_val)
    test_full = _bg(idx_test)
    train_loader = DataLoader(train_full, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_full, batch_size=bs)
    test_loader = DataLoader(test_full, batch_size=bs)

    train_occ = _bg(idx_train, forget=forget_node_indices)
    if partial:
        # Only the withdrawn records lose the charging power. The channel stays in
        # service, so the other training graphs and all validation and test graphs keep it.
        keep = ~np.isin(idx_train, withdrawn)
        train_occ = [full if k else zero for full, zero, k in zip(train_full, train_occ, keep)]
        val_occ, test_occ = val_full, test_full
    else:
        val_occ = _bg(idx_val, forget=forget_node_indices)
        test_occ = _bg(idx_test, forget=forget_node_indices)
    train_loader_occ = DataLoader(train_occ, batch_size=bs, shuffle=True)
    val_loader_occ = DataLoader(val_occ, batch_size=bs)
    test_loader_occ = DataLoader(test_occ, batch_size=bs)
    # The occlusion delta AUC always compares the test set with and without the power
    test_loader_zero = test_loader_occ if not partial else DataLoader(
        _bg(idx_test, forget=forget_node_indices), batch_size=bs)

    train_loader_noshuffle = DataLoader(train_full, batch_size=bs, shuffle=False)
    train_loader_occ_noshuffle = DataLoader(train_occ, batch_size=bs, shuffle=False)
    # Training data before this request, for the gradient-based methods: that of
    # the previous step in a sequential run, the full training set otherwise
    train_loader_before = train_loader_noshuffle
    if sequential and step > 1:
        train_loader_before = DataLoader(_bg(idx_train, forget=forget_node_indices[:step - 1]),
                                         batch_size=bs, shuffle=False)

    results = []
    all_epoch_logs = {}

    # Inputs of the IDEA noise bound, for GIF arms with noise_std: cert
    n_graph_nodes = influenced_node_count(edge_index, forget_node_indices,
                                          config.get('idea_hops', 2))
    cert = {'n_train_nodes': len(idx_train) * n_nodes,
            'n_influenced': (len(withdrawn) if partial else len(idx_train)) * n_graph_nodes,
            'eps': config.get('idea_eps', 1.0), 'delta': config.get('idea_delta', 1e-4),
            'l': config.get('idea_l', 0.25), 'lam': config.get('idea_lambda', 1.0),
            'c': config.get('idea_c', 0.5)}

    # Per-graph logits of every method, the validation curve of its fine-tuning,
    # and the parameters of the models that the cache does not hold. The
    # extra passes draw from the torch RNG like any DataLoader pass, so they run
    # after a method is scored and before reseed_arms reseeds the next one.
    outputs = {'idx_train': np.asarray(idx_train), 'idx_val': np.asarray(idx_val),
               'idx_test': np.asarray(idx_test), 'logits': {}, 'ft_curve': {}}
    if partial:
        outputs['idx_withdrawn'] = withdrawn
    in_cache = set()
    if cache is not None:
        in_cache = {'Retrain', 'Retrain_b'} | ({'Original'} if use_original_cache else set())
    if save_outputs:
        out_root = Path(config['outputs_dir'])
        (out_root / 'models').mkdir(parents=True, exist_ok=True)

    def _logits(model, loader):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(b.to(device)).cpu() for b in loader]).numpy()

    def _keep(name, model, diag=None, native=False):
        if not save_outputs:
            return
        out = {'train_occ': _logits(model, train_loader_occ_noshuffle),
               'val_occ': _logits(model, val_loader_occ),
               'test_occ': _logits(model, test_loader_occ)}
        if native:  # Original also on the inputs it was trained on
            out['train'] = _logits(model, train_loader_noshuffle)
            out['test'] = _logits(model, test_loader)
        outputs['logits'][name] = out
        if diag and diag.get('FT_curve'):
            outputs['ft_curve'][name] = np.asarray(diag['FT_curve'], dtype=float)
        if name not in in_cache:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       out_root / 'models' / f'{backbone_name}_{out_tag}_{seed}_{name}.pt')

    def _make_model():
        return build_model(backbone_name, config, data_dict, device, in_dim)

    def _eval(model, native_test_loader):
        """Compute L1 utility metrics for a given model."""
        return evaluate_model(model, native_test_loader, device)

    def _occ(model):
        if not config.get('occlusion_auc', False):
            return None
        return occlusion_delta_auc(model, test_loader, test_loader_zero, device, forget_label_idx)

    def _train(model, tr_loader, va_loader):
        return train_model_joint(
            model, tr_loader, va_loader, device,
            epochs=config['epochs'], lr=lr, weight_decay=config['weight_decay'],
            patience=config['patience'], pos_weights=pw, type_weights=tw,
            gamma=gamma, scheduler_patience=config['scheduler_patience'],
            min_delta=config.get('min_delta', 0.0))

    def _trained(kind, tr_loader, va_loader, scenario=None, use_cache=True):
        """Train a model from scratch, or load it from the cache."""
        model = _make_model()
        entry = cache.load(kind, backbone_name, seed, scenario, device) \
            if (cache is not None and use_cache) else None
        if entry is not None:
            model.load_state_dict(entry['state_dict'])
            print(f"  [{kind}] loaded from the cache")
            return model, entry['logs'], entry['time'], entry['mem']
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        model, _, logs = _train(model, tr_loader, va_loader)
        elapsed = time.time() - t0
        mem = _peak_memory_mb(device)
        if cache is not None and use_cache:
            cache.save(kind, backbone_name, seed, scenario, model, logs, elapsed, mem)
        return model, logs, elapsed, mem

    def _report(name, res, mia, elapsed, mem):
        mia_text = f"MIA_f={mia['mia_forget']:.4f}  " if not np.isnan(mia['mia_forget']) else ''
        print(f"  [{name}]  ExMatch={res['exact_match']:.4f}  "
              f"MacroROC={res['macro_roc']:.4f}  MacroF1={res['macro_f1']:.4f}  "
              f"{mia_text}Time={elapsed:.1f}s  Mem={mem:.1f}MB")

    starts = {}  # starting point of every method in a sequential run (E4)

    def _start(name):
        """theta*, or in a sequential run after S1 the model this method left after the previous step."""
        model = copy.deepcopy(model_orig)
        if sequential and step > 1:
            prev = f'S{step - 1}_seq'
            path = out_root / 'models' / f'{backbone_name}_{prev}_{seed}_{name}.pt'
            if not path.exists():
                raise FileNotFoundError(f"sequential {scen_key} starts {name} from {path}, "
                                        f"which does not exist; run S{step - 1} first")
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            starts[name] = (prev, state_hash(model))
        elif sequential:
            starts[name] = ('Original', state_hash(model))
        return model

    # ── Original: joint training on full data ──
    model_orig, logs_orig, time_orig, mem_orig = _trained(
        'Original', train_loader, val_loader, use_cache=use_original_cache)
    all_epoch_logs['Original'] = logs_orig
    res_o = _eval(model_orig, test_loader)
    mia_orig = {'mia_forget': np.nan, 'mia_retain': np.nan, 'mia_overall': np.nan}
    _report('Original', res_o, mia_orig, time_orig, mem_orig)
    results.append(_build_result(
        backbone_name, scen_key, seed, 'Original',
        res_o, mia_orig, time_orig, mem_orig,
        forget_label_idx=forget_label_idx,
        epoch_stats=_epoch_stats(logs_orig, config['epochs']), occlusion=_occ(model_orig)))
    _keep('Original', model_orig, native=True)

    # ── Unlearning methods, ablation arms, and Retrain ──
    times = {}
    for name in methods:
        if name in ('Retrain', 'Retrain_b', 'WarmStart'):
            # Reseeded in every protocol, as in the submitted code. Retrain_b
            # differs from Retrain only in the seed of the initialization and
            # of the batch order; WarmStart starts from the original model.
            offset = config.get('retrain_b_seed_offset', 1000) if name == 'Retrain_b' else 0
            _seed_all(seed + offset)
            if name == 'WarmStart':
                model_rt = _start(name)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                model_rt, _, logs_rt = _train(model_rt, train_loader_occ, val_loader_occ)
                elapsed = time.time() - t0
                mem = _peak_memory_mb(device)
            else:
                model_rt, logs_rt, elapsed, mem = _trained(
                    name, train_loader_occ, val_loader_occ, scenario=scen_tag)
            all_epoch_logs[name] = logs_rt
            res = _eval(model_rt, test_loader_occ)
            mia = compute_mia_auc(model_rt, train_loader_occ, test_loader_occ,
                                  device, pw, forget_label_idx=forget_label_idx)
            _report(name, res, mia, elapsed, mem)
            results.append(_build_result(
                backbone_name, scen_key, seed, name, res, mia, elapsed, mem,
                forget_label_idx=forget_label_idx,
                epoch_stats=_epoch_stats(logs_rt, config['epochs']), occlusion=_occ(model_rt)))
            times[name] = elapsed
            _keep(name, model_rt)
            del model_rt
            continue

        if reseed:
            _seed_all(seed)
        if ((config.get('arms') or {}).get(name) or {}).get('init', 'original') == 'scratch':
            # From-scratch control (E5): the arm's fine-tuning from a freshly
            # initialized model instead of the original one
            model = _make_model()
            if sequential:
                starts[name] = ('scratch', state_hash(model))
        else:
            model = _start(name)
        model.freeze_aux()
        diag = {}
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        if name == 'GIF':
            model = gif_unlearn(
                model, train_loader_before, train_loader_occ_noshuffle,
                criterion_weighted, device,
                iteration=config.get('gif_iteration', 50),
                damp=config.get('gif_damp', 0.01),
                scale=config.get('gif_scale', 50.0),
                max_batches=config.get('gif_max_batches'),
                grad_mode=config.get('gif_grad_mode', 'eval'),
                bn_recal=config.get('gif_bn_recal', 'cumulative'),
                hvp=config.get('gif_hvp', 'per_batch'),
                diagnostics=diag)
        elif name == 'IDEA':
            model = idea_unlearn(
                model, train_loader_before, train_loader_occ_noshuffle,
                val_loader_occ, criterion_weighted, device,
                iteration=config.get('gif_iteration', 50),
                damp=config.get('gif_damp', 0.01),
                scale=config.get('gif_scale', 50.0),
                finetune_epochs=config.get('idea_finetune', 25),
                finetune_lr=config.get('gdgu_finetune_lr', 1e-4),
                pos_weights=pw,
                max_batches=config.get('gif_max_batches'),
                grad_mode=config.get('gif_grad_mode', 'eval'),
                bn_recal=config.get('gif_bn_recal', 'cumulative'),
                hvp=config.get('gif_hvp', 'per_batch'),
                diagnostics=diag)
        elif name != 'GDGU' and (config['arms'][name] or {}).get('base') == 'gif':
            model = gif_arm_unlearn(
                model, train_loader_before, train_loader_occ_noshuffle,
                val_loader_occ, criterion_weighted, device, pos_weights=pw,
                cert=cert, diagnostics=diag, **gif_kwargs(config, config['arms'][name]))
        else:
            overrides = None if name == 'GDGU' else config['arms'][name]
            model = gdgu_feature_unlearn(
                model, train_loader_before, train_loader_occ_noshuffle,
                val_loader_occ, criterion_weighted, device, pos_weights=pw,
                diagnostics=diag, **gdgu_kwargs(config, overrides))
        elapsed = time.time() - t0
        mem = _peak_memory_mb(device)
        res = _eval(model, test_loader_occ)
        mia = compute_mia_auc(model, train_loader_occ, test_loader_occ,
                              device, pw, forget_label_idx=forget_label_idx)
        _report(name, res, mia, elapsed, mem)
        results.append(_build_result(
            backbone_name, scen_key, seed, name, res, mia, elapsed, mem,
            forget_label_idx=forget_label_idx, diagnostics=diag, occlusion=_occ(model)))
        times[name] = elapsed
        _keep(name, model, diag)
        del model

    if 'Retrain' in times:
        speedups = '  '.join(f"{n} {times['Retrain'] / max(t, 1e-6):.1f}x"
                             for n, t in times.items() if n != 'Retrain')
        print(f"  Speedup vs Retrain: {speedups}")

    if save_outputs:
        torch.save(outputs, out_root / f'{backbone_name}_{out_tag}_{seed}.pt')

    del model_orig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if v_noise is not None:
        for row in results:
            row.update(v_noise)
    if partial:
        for row in results:
            row.update({'Withdraw_frac': frac, 'N_withdrawn': len(withdrawn)})
    if holdout is not None:
        for row in results:
            row['Holdout_type'] = holdout
    if semantics != 'zero_std':
        for row in results:
            row['Forget_semantics'] = semantics
    if sequential:
        for row in results:
            start, digest = starts.get(row['Method'], ('scratch', ''))
            row.update({'Sequential': True, 'Seq_start': start, 'Start_hash': digest})
    return results, all_epoch_logs
