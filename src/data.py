"""Data loading, graph construction, and splitting for EVCS localization."""

import gc
import pickle
import gzip
import numpy as np
import torch
import networkx as nx
from collections import Counter
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Pre-defined EVCS-to-bus mappings
EVCS_PRESETS = {
    '34bus': {
        # Order matches PowerBench source DataGeneration.py StationsInfo:
        # EVCS 1 -> bus 814, EVCS 2 -> bus 890, EVCS 3 -> bus 852.
        'EVCS 1': '814',
        'EVCS 2': '890',
        'EVCS 3': '852',
    },
    '123bus': {
        'EVCS 1': '25',
        'EVCS 2': '40',
        'EVCS 3': '54',
        'EVCS 4': '62',
        'EVCS 5': '76',
    },
    '8500bus': {
        'EVCS 1': 'l3104136',
        'EVCS 2': 'l2895449',
        'EVCS 3': 'l3010560',
        'EVCS 4': 'l2876797',
        'EVCS 5': 'l2876814',
        'EVCS 6': 'l3081380',
        'EVCS 7': 'l2766718',
    },
}


def load_evcs_data(pkl_paths, gml_path, evcs_map=None, bus_system='34bus',
                   feature_mode='mean_std'):
    """Load raw EVCS attack data and build node features, multi-hot labels, edge_index.

    Args:
        pkl_paths:    str or list of str. Single .pkl file or list of .pkl.gz files.
        gml_path:     str. Path to GML topology file.
        evcs_map:     dict or None. {'EVCS 1': '814', ...}. If None, uses preset.
        bus_system:   str. '34bus' or '123bus', used for preset lookup and returned tag.
        feature_mode: 'mean_std' (default) — 24 hourly mean + 24 hourly std = 48-dim;
                      'raw' — phase-averaged voltage at all 288 timesteps = 288-dim.

    Returns dict with keys:
        all_x, all_y, edge_index, n_nodes, n_feat, n_evcs,
        bus_names, bus_to_idx, evcs_buses, evcs_map, evcs_names, bus_system
    """
    # Resolve EVCS mapping
    if evcs_map is None:
        evcs_map = EVCS_PRESETS[bus_system]
    evcs_names_ordered = sorted(evcs_map.keys(), key=lambda k: int(k.split()[-1]))
    n_evcs = len(evcs_names_ordered)

    # Stream pkl files one at a time to avoid holding all raw Python dicts
    # in memory simultaneously (each 6.1GB pkl.gz expands to ~70GB of Python
    # objects, so a single batch load of all 4 pkls peaks at ~280GB).
    if isinstance(pkl_paths, str):
        pkl_paths = [pkl_paths]

    chunks_x = []
    chunks_y = []
    bus_names = None
    bus_to_idx = None

    for p in pkl_paths:
        opener = gzip.open if p.endswith('.gz') else open
        raw_chunk = []
        with opener(p, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, list):
                raw_chunk.extend(obj)
            else:
                raw_chunk.append(obj)
                try:
                    while True:
                        raw_chunk.append(pickle.load(f))
                except EOFError:
                    pass

        # First pkl: extract bus names and index map
        if bus_names is None:
            bus_names = list(raw_chunk[0]['BusVoltage series'].keys())
            bus_to_idx = {name: i for i, name in enumerate(bus_names)}

        n_g = len(raw_chunk)
        n_nodes = len(bus_names)

        n_feat_v = 288 if feature_mode == 'raw' else 48
        x = np.zeros((n_g, n_nodes, n_feat_v), dtype=np.float32)
        y = np.zeros((n_g, n_evcs), dtype=np.float32)

        for gi, scenario in enumerate(raw_chunk):
            bv = scenario['BusVoltage series']
            for ni, bus in enumerate(bus_names):
                voltages = np.array(bv[bus])  # (288, 3)
                # Use only active phases (>0.1 p.u.) to avoid 3x scaling artifact
                # on single-phase buses (810, 818, 820, 822, 826, 856, 864, 838).
                active = voltages.mean(axis=0) > 0.1
                if active.any():
                    mean_phase = voltages[:, active].mean(axis=1)  # (288,)
                else:
                    mean_phase = voltages.mean(axis=1)
                if feature_mode == 'raw':
                    x[gi, ni, :] = mean_phase
                else:
                    hourly = mean_phase.reshape(24, 12)
                    x[gi, ni, :24] = hourly.mean(axis=1)
                    x[gi, ni, 24:] = hourly.std(axis=1)

            targeted = scenario['Targeted Stations']
            for evcs_idx, evcs_name in enumerate(evcs_names_ordered):
                if evcs_name in targeted:
                    y[gi, evcs_idx] = 1.0

        chunks_x.append(x)
        chunks_y.append(y)

        # Free this pkl's raw Python objects before loading the next file
        del raw_chunk
        gc.collect()

    all_x = np.concatenate(chunks_x, axis=0)
    all_y = np.concatenate(chunks_y, axis=0)
    del chunks_x, chunks_y
    gc.collect()

    n_graphs = all_x.shape[0]
    n_feat = all_x.shape[2]
    n_nodes = all_x.shape[1]

    # EVCS bus indices
    evcs_buses = {}
    for evcs_name in evcs_names_ordered:
        bus_id = evcs_map[evcs_name]
        try:
            key = int(bus_id)
        except ValueError:
            key = bus_id
        evcs_buses[key] = bus_to_idx[bus_id]

    # Edge index from GML topology (bidirectional)
    G = nx.read_gml(gml_path)
    edges_src, edges_dst = [], []
    gml_to_idx = {n: bus_to_idx.get(n, -1) for n in G.nodes()}
    for u, v in G.edges():
        ui, vi = gml_to_idx.get(u, -1), gml_to_idx.get(v, -1)
        if ui >= 0 and vi >= 0:
            edges_src.extend([ui, vi])
            edges_dst.extend([vi, ui])
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Summary
    n_normal = int((all_y.sum(axis=1) == 0).sum())
    print(f"[{bus_system}] Graphs: {n_graphs}")
    print(f"  Nodes  : {n_nodes}, Features: {n_feat}")
    print(f"  Edges  : {edge_index.shape[1]} (bidirectional)")
    print(f"  Labels : Normal={n_normal}, Attack={n_graphs - n_normal}")
    print(f"  EVCS   : {n_evcs} stations")
    print(f"\n  Per-EVCS attack frequency:")
    for i, evcs_name in enumerate(evcs_names_ordered):
        bus_id = evcs_map[evcs_name]
        print(f"    {evcs_name} (Bus {bus_id}): {int(all_y[:, i].sum())}/{n_graphs}")
    combos = Counter([tuple(row) for row in all_y.astype(int)])
    print(f"\n  Multi-label distribution:")
    for combo, cnt in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"    {list(combo)}: {cnt}")

    return {
        'all_x': all_x,
        'all_y': all_y,
        'edge_index': edge_index,
        'n_graphs': n_graphs,
        'n_nodes': n_nodes,
        'n_feat': n_feat,
        'n_evcs': n_evcs,
        'bus_names': bus_names,
        'bus_to_idx': bus_to_idx,
        'evcs_buses': evcs_buses,
        'evcs_map': evcs_map,
        'evcs_names': evcs_names_ordered,
        'bus_system': bus_system,
    }


def stratified_split_multilabel(y, test_size, val_ratio, seed):
    """Stratified split for multi-label.

    Tries hash-based stratify (one class per unique label vector) first; on
    8500-bus the 7-EVCS label space yields up to 128 combinations and many
    have only 1 sample, so sklearn rejects the strict split. Fall back to
    stratifying by attack count (sum across columns), which collapses 128
    classes into 8 (0-7 attacked EVCS) and stays robust on sparse data.
    """
    idx_all = np.arange(len(y))
    label_hash = np.array([hash(tuple(row)) for row in y])
    try:
        idx_train, idx_temp = train_test_split(
            idx_all, test_size=test_size, stratify=label_hash, random_state=seed)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=val_ratio, stratify=label_hash[idx_temp], random_state=seed)
    except ValueError:
        label_sum = y.sum(axis=1).astype(int)
        idx_train, idx_temp = train_test_split(
            idx_all, test_size=test_size, stratify=label_sum, random_state=seed)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=val_ratio, stratify=label_sum[idx_temp], random_state=seed)
    return idx_train, idx_val, idx_test


def fit_scaler(all_x, idx_train):
    """Fit StandardScaler on training data (flattened node features)."""
    x_train_flat = all_x[idx_train].reshape(len(idx_train), -1)
    return StandardScaler().fit(x_train_flat)


# ======================================================================
#  A-append data pipeline (V | P concatenated node features)
# ======================================================================

TYPE_MAP = {"Nil": 0, "Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}


def augment_with_power(data_dict, pkl_paths, feature_mode='mean_std'):
    """Extend load_evcs_data output with P features and attack-type labels.

    Adds keys: all_x_P [G, N, n_feat], all_attack_type [G], evcs_node_indices [K].
    Streams pkl files one at a time to bound peak memory.
    feature_mode: 'mean_std' (48-dim) or 'raw' (288-dim), must match load_evcs_data.
    """
    if isinstance(pkl_paths, str):
        pkl_paths = [pkl_paths]

    evcs_node_indices = [
        data_dict['bus_to_idx'][data_dict['evcs_map'][nm]]
        for nm in data_dict['evcs_names']
    ]
    n_nodes = data_dict['n_nodes']

    chunks_x_P = []
    chunks_attack_type = []

    for p in pkl_paths:
        opener = gzip.open if p.endswith('.gz') else open
        raw_chunk = []
        with opener(p, 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, list):
                raw_chunk.extend(obj)
            else:
                raw_chunk.append(obj)
                try:
                    while True:
                        raw_chunk.append(pickle.load(f))
                except EOFError:
                    pass

        n_g = len(raw_chunk)
        n_feat_p = 288 if feature_mode == 'raw' else 48
        x_P = np.zeros((n_g, n_nodes, n_feat_p), dtype=np.float32)
        attack_types = np.zeros(n_g, dtype=np.int64)

        for gi, s in enumerate(raw_chunk):
            for evcs_1based, p_series in s["EVCS power series"].items():
                idx0 = evcs_1based - 1
                node_idx = evcs_node_indices[idx0]
                p_arr = np.asarray(p_series, dtype=np.float32)  # (288,)
                if feature_mode == 'raw':
                    x_P[gi, node_idx, :] = p_arr
                else:
                    p_hourly = p_arr.reshape(24, 12)
                    x_P[gi, node_idx, :24] = p_hourly.mean(axis=1)
                    x_P[gi, node_idx, 24:] = p_hourly.std(axis=1)
            attack_types[gi] = TYPE_MAP[s["Attack Type"]]

        chunks_x_P.append(x_P)
        chunks_attack_type.append(attack_types)
        del raw_chunk
        gc.collect()

    all_x_P = np.concatenate(chunks_x_P, axis=0)
    all_attack_type = np.concatenate(chunks_attack_type, axis=0)
    del chunks_x_P, chunks_attack_type
    gc.collect()

    data_dict['all_x_P'] = all_x_P
    data_dict['all_attack_type'] = all_attack_type
    data_dict['evcs_node_indices'] = evcs_node_indices
    return data_dict


def build_graphs(all_x_V, all_x_P, all_y, all_attack_type,
                          edge_index, n_nodes, idx,
                          scaler_V, scaler_P,
                          forget_P_at_node=None):
    """Build PyG Data list (A-append [V | P], optional P zeroing).

    Args:
        forget_P_at_node: if int, zero dims 48:96 at this node for every graph.
    """
    n = len(idx)
    xV = all_x_V[idx]
    xP = all_x_P[idx]
    n_feat = all_x_V.shape[2]  # per-modality feature dim (48 mean_std / 288 raw)
    v = scaler_V.transform(xV.reshape(n, -1)).reshape(n, n_nodes, n_feat).astype(np.float32)
    p = scaler_P.transform(xP.reshape(n, -1)).reshape(n, n_nodes, n_feat).astype(np.float32)
    x = np.concatenate([v, p], axis=2)

    if forget_P_at_node is not None:
        x[:, forget_P_at_node, n_feat:] = 0.0

    ds = []
    for i in range(n):
        ds.append(Data(
            x=torch.tensor(x[i], dtype=torch.float),
            edge_index=edge_index.clone(),
            y=torch.tensor(all_y[idx[i]], dtype=torch.float).unsqueeze(0),
            y_type=torch.tensor([all_attack_type[idx[i]]], dtype=torch.long),
        ))
    return ds
