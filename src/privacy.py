"""src/privacy.py — V6.0 Route A privacy evaluation suite (v2).

L2 modality-erasure metrics + unlearn efficiency helper.

All metrics assume a single-head localization model with:
  - ``forward(data) -> logits [B, K]``
  - ``encode(data) -> [B, 2*hid_dim]`` pooled graph embedding
  - ``convs``, ``bns`` ModuleLists (per the GCN_Graph / GAT_Graph / GIN_Graph
    convention in src/models.py).

Graphs are assumed to have constant node count (true for 34-bus / 123-bus
where every graph covers the full feeder).

Sub-metrics (higher number = stronger evidence model still leverages P):
  L2-a  Integrated-Gradients attribution on P dims at forget node
  L2-b  Occlusion Delta-AUC — PRIMARY (Macro-ROC drop when P zeroed)
  L2-c  Reconstruction attack: decoder(graph_emb) -> low-dim P target
  L2-e  Attack-type inference: 5-way linear probe on graph embedding

v1 -> v2 changes:
  - L2-c switched from forget-node emb (128d, saturated by message-passing)
    to pooled graph emb (256d); target reduced from P_48 to caller-chosen dim
  - L2-d removed (node-level embedding collapse made probe unreliable)
  - L2-e switched from forget-node emb to pooled graph emb (attack type is a
    graph-level label, single-node emb cannot resolve it)
  - L2-b promoted to PRIMARY (clearest directional signal in smoke validation)

Efficiency:
  measure_unlearn_efficiency(fn, ...) wraps any callable and returns
  wall-clock seconds alongside its result.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


__all__ = [
    "derive_det_from_loc",
    "extract_forget_node_embeddings",
    "extract_graph_embeddings",
    "L2_a_integrated_gradients",
    "L2_b_occlusion_delta_auc",
    "L2_c_reconstruction",
    "L2_e_attack_type_inference",
    "measure_unlearn_efficiency",
]


# =====================================================================
#  Derived detection from localization logits
# =====================================================================

def derive_det_from_loc(loc_logits: torch.Tensor) -> torch.Tensor:
    """y_det = max over K of sigmoid(y_loc_logits). Returns [B]."""
    return torch.sigmoid(loc_logits).max(dim=-1).values


# =====================================================================
#  Node-level embedding extractor (forward hook on last BN)
# =====================================================================

def extract_forget_node_embeddings(model, loader, forget_node_idx, n_nodes, device):
    """Return [G, H] post-BN+ReLU embedding tensor at the forget node.

    Iterates over ``loader`` in eval mode, captures the output of the last BN
    layer via a forward hook, applies ReLU (to match the model's forward
    pipeline), and selects the row corresponding to ``forget_node_idx`` within
    each graph.
    """
    model.eval()
    last_bn = model.bns[-1]

    captured = {}

    def _hook(_module, _inputs, output):
        captured["x"] = output

    handle = last_bn.register_forward_hook(_hook)

    outs = []
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _ = model(batch)
                x_nodes = captured["x"]               # [sum_N, H]
                x_nodes = F.relu(x_nodes)             # match model forward
                B = batch.num_graphs
                H = x_nodes.size(-1)
                x_3d = x_nodes.view(B, n_nodes, H)
                outs.append(x_3d[:, forget_node_idx, :].detach().cpu())
    finally:
        handle.remove()

    return torch.cat(outs, dim=0)                     # [G, H]


# =====================================================================
#  Graph-level embedding extractor (via model.encode)
# =====================================================================

def extract_graph_embeddings(model, loader, device):
    """Return [G, D] pooled graph embeddings via ``model.encode(data)``."""
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.encode(batch)                 # [B, 2*hid_dim]
            outs.append(emb.detach().cpu())
    return torch.cat(outs, dim=0)                     # [G, D]


# =====================================================================
#  L2-a: Integrated Gradients attribution
# =====================================================================

def L2_a_integrated_gradients(
    model,
    loader,
    forget_node_idx,
    forget_evcs_k,
    n_nodes,
    device,
    n_steps: int = 50,
    baseline: str = "zero",
):
    """Integrated Gradients of ``logits[:, forget_evcs_k]`` w.r.t. P dims at
    ``forget_node_idx``.

    IG solves the loss-gradient saturation artifact at F1=1.0 (loss ~ 0 ->
    grad ~ 0) by using the logit directly as the target scalar.

    Returns mean / std of the per-graph sum of |IG| across the 48 P dims.
    """
    model.eval()
    vals = []
    for batch in loader:
        batch = batch.to(device)
        x_orig = batch.x.detach().clone()
        x_base = torch.zeros_like(x_orig) if baseline == "zero" else x_orig.mean(0, keepdim=True).expand_as(x_orig)

        accum = torch.zeros_like(x_orig)
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps         # midpoint rule
            x_interp = (x_base + alpha * (x_orig - x_base)).detach().requires_grad_(True)
            batch.x = x_interp
            logits = model(batch)                   # [B, K]
            target = logits[:, forget_evcs_k].sum()
            grad = torch.autograd.grad(target, x_interp)[0]
            accum = accum + grad.detach()

        avg_grad = accum / n_steps
        ig_full = avg_grad * (x_orig - x_base)      # [sum_N, 2*n_feat]
        B = batch.num_graphs
        ig_3d = ig_full.view(B, n_nodes, -1)        # [B, N, 2*n_feat] (V|P)
        n_feat = ig_3d.shape[-1] // 2                # P modality starts at half
        sel = ig_3d[:, forget_node_idx, n_feat:]     # [B, (n_forget), n_feat]
        ig_p = sel.abs().reshape(B, -1).sum(dim=-1)  # [B]
        vals.append(ig_p.detach().cpu().numpy())

        batch.x = x_orig                            # restore for safety

    vals = np.concatenate(vals)
    return {"mean": float(vals.mean()), "std": float(vals.std())}


# =====================================================================
#  L2-b: Occlusion Delta-AUC
# =====================================================================

def _eval_macro_roc(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            ys.append(batch.y.view(-1, logits.size(-1)).cpu())
            ps.append(torch.sigmoid(logits).cpu())
    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(ps, dim=0).numpy()
    try:
        return float(roc_auc_score(y, p, average="macro"))
    except ValueError:
        return float("nan")


def L2_b_occlusion_delta_auc(model, loader_P_present, loader_P_occluded, device):
    """Macro-ROC drop when the forget node's P channel is zeroed at test.

    Small delta (~0) indicates the model no longer relies on the forgotten P.
    """
    auc_p = _eval_macro_roc(model, loader_P_present, device)
    auc_o = _eval_macro_roc(model, loader_P_occluded, device)
    return {
        "auc_P_present": auc_p,
        "auc_P_occluded": auc_o,
        "delta_auc": auc_p - auc_o,
    }


# =====================================================================
#  L2-c: Reconstruction attack from graph embedding
# =====================================================================

def L2_c_reconstruction(
    model,
    train_loader,
    test_loader,
    device,
    target_extractor,
    epochs: int = 40,
    lr: float = 1e-3,
    hidden: int = 128,
):
    """Train a small MLP decoder from pooled graph embedding to a continuous
    target (e.g. low-dim P summary at the forget node).

    Args:
        target_extractor:  callable ``(batch) -> [B, T]`` giving the target
                           tensor to reconstruct. The caller chooses what to
                           reconstruct (and its dimensionality).

    Returns MSE on test vs. a population-mean baseline and their ratio.
    """
    tr_emb = extract_graph_embeddings(model, train_loader, device)
    te_emb = extract_graph_embeddings(model, test_loader,  device)

    def _gather(loader):
        ts = []
        for batch in loader:
            batch = batch.to(device)
            ts.append(target_extractor(batch).detach().cpu())
        return torch.cat(ts, dim=0)

    tr_t = _gather(train_loader)
    te_t = _gather(test_loader)

    H = tr_emb.size(1)
    T = tr_t.size(1)
    decoder = nn.Sequential(
        nn.Linear(H, hidden), nn.ReLU(),
        nn.Linear(hidden, T),
    ).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    tr_emb_d = tr_emb.to(device)
    tr_t_d = tr_t.to(device)
    for _ in range(epochs):
        decoder.train()
        opt.zero_grad()
        pred = decoder(tr_emb_d)
        loss = F.mse_loss(pred, tr_t_d)
        loss.backward()
        opt.step()

    decoder.eval()
    with torch.no_grad():
        pred_te = decoder(te_emb.to(device)).cpu()
        mse_te = float(F.mse_loss(pred_te, te_t).item())
        mean = tr_t.mean(dim=0, keepdim=True).expand_as(te_t)
        mse_baseline = float(F.mse_loss(mean, te_t).item())

    return {
        "mse_test": mse_te,
        "mse_baseline": mse_baseline,
        "mse_ratio": mse_te / max(mse_baseline, 1e-12),
    }


# =====================================================================
#  L2-e: Attack-type linear-probe on graph embedding
# =====================================================================

def _train_linear_probe(
    train_feat, train_lab, n_classes, device,
    epochs=60, lr=1e-3, weight_decay=1e-4,
):
    H = train_feat.size(1)
    probe = nn.Linear(H, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    feat_d = train_feat.to(device)
    lab_d = train_lab.to(device).long()
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(feat_d)
        loss = crit(logits, lab_d)
        loss.backward()
        opt.step()
    return probe


def _probe_accuracy(probe, test_feat, test_lab, device):
    probe.eval()
    with torch.no_grad():
        logits = probe(test_feat.to(device))
        preds = logits.argmax(dim=-1).cpu()
    return float((preds == test_lab.long()).float().mean().item())


def L2_e_attack_type_inference(
    model,
    train_loader, test_loader,
    train_labels, test_labels,
    n_classes, device, epochs=60,
):
    """5-way (nil + 4 attack types) linear probe on frozen pooled graph
    embedding from ``model.encode()``.

    Three-point interpretation:
      - Original:  acc ~ ceiling (model sees P directly)
      - Unlearned: acc should approach Retrain-A
      - Retrain-A: physical-leakage floor (what V alone reveals)
    """
    tr = extract_graph_embeddings(model, train_loader, device)
    te = extract_graph_embeddings(model, test_loader,  device)
    probe = _train_linear_probe(tr, train_labels, n_classes, device, epochs=epochs)
    acc = _probe_accuracy(probe, te, test_labels, device)
    return {"test_accuracy": acc, "chance_level": 1.0 / n_classes}


# =====================================================================
#  Efficiency measurement
# =====================================================================

def measure_unlearn_efficiency(unlearn_fn, *args, **kwargs):
    """Call ``unlearn_fn(*args, **kwargs)`` and return (result, elapsed_s)."""
    t0 = time.perf_counter()
    result = unlearn_fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0)
