"""Training, evaluation, and checkpoint utilities for multi-label GNN."""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss


def kaiming_init(model):
    """Kaiming uniform initialization for ReLU-based networks.

    Kaiming compensates for ReLU zeroing ~50% of activations, giving ~41%
    larger initial weights than Xavier for the same layer size.  This improves
    convergence on weak-signal data (e.g. feature-zeroed unlearning graphs).
    """
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            nn.init.kaiming_uniform_(param, nonlinearity='relu')
        elif 'bias' in name:
            nn.init.zeros_(param)


def get_pos_weights(y_train, device):
    """Compute pos_weight for BCEWithLogitsLoss: neg_count / pos_count per label."""
    y = np.array(y_train)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    pw = neg / np.maximum(pos, 1)
    return torch.tensor(pw, dtype=torch.float).to(device)


def _compute_val_roc(model, val_loader, device):
    """Compute macro ROC-AUC on validation set."""
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
        return 0.5
    valid_cols = [c for c in range(ys.shape[1]) if len(np.unique(ys[:, c])) > 1]
    if len(valid_cols) == 0:
        return 0.5
    return float(np.mean([roc_auc_score(ys[:, c], probs[:, c]) for c in valid_cols]))


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Multi-label evaluation.

    Returns dict with: exact_match, hamming_acc, macro_f1, macro_roc,
                       per_roc, per_f1, preds, probs, labels.
    """
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch).cpu())
        all_y.append(batch.y.cpu())

    logits = torch.cat(all_logits).numpy()
    ys = torch.cat(all_y).numpy()
    probs = 1 / (1 + np.exp(-logits))
    if np.any(np.isnan(probs)):
        probs = np.nan_to_num(probs, nan=0.5)

    preds = (probs >= 0.5).astype(float)
    exact_match = (preds == ys).all(axis=1).mean()
    h_acc = 1.0 - hamming_loss(ys, preds)

    n_labels = ys.shape[1]
    per_roc, per_f1 = [], []
    for c in range(n_labels):
        if len(np.unique(ys[:, c])) > 1:
            per_roc.append(roc_auc_score(ys[:, c], probs[:, c]))
            per_f1.append(f1_score(ys[:, c], preds[:, c]))
        else:
            per_roc.append(0.5)
            per_f1.append(0.0)

    return {
        'exact_match': exact_match,
        'hamming_acc': h_acc,
        'macro_f1': float(np.mean(per_f1)),
        'macro_roc': float(np.mean(per_roc)),
        'per_roc': per_roc,
        'per_f1': per_f1,
        'preds': preds,
        'probs': probs,
        'labels': ys,
    }


# ======================================================================
#  V6.0 Route A: joint training (loc + aux attack-type) + aux eval
# ======================================================================

def train_model_joint(model, train_loader, val_loader, device,
                      epochs=200, lr=5e-4, weight_decay=1e-4, patience=30,
                      pos_weights=None, type_weights=None, gamma=0.5,
                      verbose=False, scheduler_patience=20):
    """Joint training: L = BCE(loc) + gamma * CE(attack_type).

    model must expose forward_both(data) -> (loc_logits, type_logits).
    Monitors val macro-ROC on localization for early stopping.
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=scheduler_patience)
    crit_loc = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    crit_type = nn.CrossEntropyLoss(
        weight=type_weights) if type_weights is not None else nn.CrossEntropyLoss()

    best_metric = 0.0
    best_state = None
    wait = 0
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loc_logits, type_logits = model.forward_both(batch)
            loss = crit_loc(loc_logits, batch.y) + gamma * crit_type(
                type_logits, batch.y_type.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_metric = _compute_val_roc(model, val_loader, device)
        scheduler.step(val_metric)
        epoch_logs.append({
            'epoch': epoch,
            'train_loss': round(total_loss / max(n_batches, 1), 6),
            'val_roc': round(val_metric, 6),
        })

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}, best val macro-ROC={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metric, epoch_logs


@torch.no_grad()
def evaluate_aux_acc(model, loader, device):
    """Attack-type auxiliary head accuracy (5-way)."""
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model.forward_aux(batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch.y_type.view(-1)).sum().item()
        total += batch.y_type.size(0)
    return correct / max(total, 1)


def compute_mia_auc(model, member_loader, non_member_loader, device,
                    pos_weights=None, forget_label_idx=None):
    """Loss-based MIA for multi-label, split by forget/retain labels.

    Following OpenGU (Fan et al., 2025), MIA should measure information
    leakage about the *forgotten* data specifically, not the overall
    train/test generalization gap.

    Args:
        forget_label_idx: list of label column indices for forget EVCS.
            e.g. S1 forgets EVCS1 -> [0]; S2 -> [0,1]; S3 -> [0,1,2].
            If None, falls back to overall-only (backward compatible).

    Returns:
        dict with keys: mia_forget, mia_retain, mia_overall.
        mia_retain is NaN when all labels are forget (e.g. S3).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
    all_losses, all_labels = [], []  # per-label losses [B, n_labels]

    with torch.no_grad():
        for loader, lbl in [(member_loader, 1), (non_member_loader, 0)]:
            for batch in loader:
                batch = batch.to(device)
                loss = criterion(model(batch), batch.y)  # [B, n_labels]
                all_losses.append(loss.cpu())
                all_labels.append(torch.full((loss.size(0),), lbl))

    losses = torch.cat(all_losses).numpy()   # [N, n_labels]
    labels = torch.cat(all_labels).numpy()   # [N]

    if np.any(np.isnan(losses)) or len(np.unique(labels)) < 2:
        return {'mia_forget': 0.5, 'mia_retain': 0.5, 'mia_overall': 0.5}

    def _auc(col_idx):
        if not col_idx:
            return np.nan
        scores = -losses[:, col_idx].mean(axis=1)
        return float(roc_auc_score(labels, scores))

    n_labels = losses.shape[1]
    all_idx = list(range(n_labels))

    if forget_label_idx is not None:
        retain_idx = [i for i in all_idx if i not in forget_label_idx]
        mia_forget = _auc(forget_label_idx)
        mia_retain = _auc(retain_idx)
    else:
        mia_forget = np.nan
        mia_retain = np.nan

    mia_overall = _auc(all_idx)
    return {'mia_forget': mia_forget, 'mia_retain': mia_retain,
            'mia_overall': mia_overall}


def save_checkpoint(model, path, backbone, scenario, seed):
    """Save model state_dict to results/checkpoints/."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filename = os.path.join(path, f"{backbone}_{scenario}_{seed}.pth")
    torch.save(model.state_dict(), filename)
    return filename
