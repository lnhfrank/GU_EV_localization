"""E9 parts 1 and 2 (light): how far the attack information of a charging station reaches in the bus voltages.

Part 1, voltage-only predictability. For every bus i and station j, a logistic regression on bus i's
phase-averaged voltage (288 five-minute steps, the raw features of the GNN) predicts whether station j
is attacked; 5-fold stratified CV, ROC-AUC. Main labels: station j's attack label on all graphs, the
same task as the GNN leakage floor. Measurement noise as in E9 part 4 (src/noise.py), at clean, 20, 10
and 0 dB. By attack type: normal graphs against graphs where only station j is attacked, with type T.

Part 2, LinDistFlow (light). The linearized power flow gives the voltage change at bus i caused by a
power change at station j as dV_i = -R_c(i, j) dP_j, where R_c is the resistance of the path that i and
j share towards the substation (per unit, so lines behind a transformer count on their own voltage
base). Voltage regulators hold their output voltage, so only the shared part below the last regulator
on the shared path counts (R_eff), and a bus behind a regulator that the station's path does not pass
is shielded (R_eff = 0). The footprint share rho_ij = R_eff(i, j) / R_eff(j, j) is 1 at the station and
at every unshielded bus downstream of it, and falls towards 0 for buses that branch off early or sit
behind a regulator; the electrical distance is d_ij = 1 - rho_ij. rho_root, the same share without the
regulators, is kept for comparison. The check regresses the measured voltage deviation of
single-station attack graphs from the mean normal day on the deviation of the station's charging
power, bus by bus (beta_ij, p.u. per kW), and compares it with the prediction -R_eff(i, j) / 1000
(S_base = 1 MVA, so 1 kW = 1e-3 p.u.).

Writes, to results/2026-10-12_E9topo or to the folder given as the first argument: e9_distance.csv,
e9_auc_all.csv, e9_auc_type.csv, e9_lindistflow.csv and compute_e9.txt (the printed summary). The PowerBench
EVCS data are read from EVCS_DATA, as --data_dir of train.py.
"""
import glob
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]             # the repository
sys.path.insert(0, str(ROOT))
from src.noise import add_voltage_noise  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / '2026-10-12_E9topo'   # the output folder
OUT.mkdir(parents=True, exist_ok=True)
EVCS_DATA = Path(os.environ.get('EVCS_DATA', ROOT.parent / 'Source' / 'PB_data' / '3_EVCS Attacks'))
DATA_CACHE = ROOT / 'results' / '2026-10-01_E3' / 'data_cache'
GML = {'34bus': EVCS_DATA / '34_bus' / '34busEx.gml', '123bus': EVCS_DATA / '123_bus' / '123busEx.gml'}
# Nominal line-to-line voltage of each section (kV). The IEEE 34-bus feeder runs at 24.9 kV except
# the 4.16 kV section behind transformer xfm1 (832-888); the 123-bus feeder runs at 4.16 kV except the
# 0.48 kV bus behind xfm1 (61s-610). The substation transformers and regulators have tiny impedance.
LOW_SIDE = {'34bus': ({'888', '890'}, 24.9, 4.16), '123bus': ({'610'}, 4.16, 0.48)}
SNR_DB = [None, 20, 10, 0]          # None is the clean data
TYPE_SNR = [None, 10]               # noise levels for the attack-type decomposition
NOISE_SEED = 42
ROOT_BUS = {'34bus': 'sourcebus', '123bus': '150'}   # substation bus of each feeder
N_JOBS = 32


def load(bus):
    path = sorted(glob.glob(str(DATA_CACHE / f'{bus}_raw_*.pkl')))[0]
    with open(path, 'rb') as f:
        return pickle.load(f)


def per_unit_tree(bus):
    """Undirected GML tree with each edge's resistance and reactance in per unit (S_base cancels)."""
    G = nx.read_gml(GML[bus])
    low, kv_hi, kv_lo = LOW_SIDE[bus]
    for u, v, a in G.edges(data=True):
        # An edge belongs to the low-voltage section when it touches it; transformer xfm1 is given in
        # ohms on its low-voltage side (34-bus: 1.41 ohm reactance = 4.08% on 500 kVA, 4.16 kV)
        kv = kv_lo if (u in low or v in low) else kv_hi
        zbase = kv ** 2                 # (kV)^2 / MVA with S_base = 1 MVA
        a['r_pu'] = a['Resistance'] / zbase
        a['x_pu'] = a['Reactance'] / zbase
    return G


def footprint(G, buses, stations, root):
    """Per (bus, station): shared resistance with and without regulators, the share rho, hops, category."""
    path = nx.shortest_path(G, root)                       # root-to-node node lists
    is_reg = lambda u, v: str(G[u][v].get('Label', '')).startswith('reg')
    edges = lambda p: list(zip(p[:-1], p[1:]))

    def shared(i, j):
        pi, pj = path[i], path[j]
        k = 0
        while k < min(len(pi), len(pj)) and pi[k] == pj[k]:
            k += 1
        return edges(pi[:k]), edges(pi[k - 1:])              # shared root path, bus i's own branch

    def resist(i, j):
        sh, own = shared(i, j)
        r_root = sum(G[u][v]['r_pu'] for u, v in sh)
        last = max([t for t, (u, v) in enumerate(sh) if is_reg(u, v)], default=-1)
        shielded = any(is_reg(u, v) for u, v in own)
        r_eff = 0.0 if shielded else sum(G[u][v]['r_pu'] for u, v in sh[last + 1:])
        cat = ('shielded by a regulator' if shielded else
               'regulator on the shared path' if last >= 0 else 'no regulator')
        return r_root, r_eff, cat

    rows = []
    for j in stations:
        rj_root, rj_eff, _ = resist(j, j)
        for i in buses:
            r_root, r_eff, cat = resist(i, j)
            rows.append(dict(station_bus=j, bus=i, r_root_pu=r_root, r_eff_pu=r_eff,
                             rho_root=r_root / rj_root, rho=r_eff / rj_eff, d_elec=1 - r_eff / rj_eff,
                             hops=nx.shortest_path_length(G, i, j), regulator_category=cat))
    return pd.DataFrame(rows)


def cv_auc(X, y, seed=42):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)   # inside the worker process
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced',
                                                             solver='lbfgs', C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, proba)


def part1(bus, data, log):
    X_all, Y, T = data['all_x'], data['all_y'], data['all_attack_type']
    names, st_idx = data['bus_names'], data['evcs_node_indices']
    st_names = [data['evcs_map'][e] for e in data['evcs_names']]
    rows_all, rows_type = [], []
    for snr in SNR_DB:
        X = X_all if snr is None else add_voltage_noise(X_all, snr, NOISE_SEED)[0]
        t0 = time.time()
        res = Parallel(n_jobs=N_JOBS)(delayed(cv_auc)(X[:, i, :], Y[:, j])
                                      for j in range(len(st_idx)) for i in range(len(names)))
        k = 0
        for j in range(len(st_idx)):
            for i in range(len(names)):
                rows_all.append(dict(feeder=bus, station=data['evcs_names'][j], station_bus=st_names[j],
                                     bus=names[i], snr_db=snr, auc=res[k])); k += 1
        log(f'  {bus} part 1, all graphs, SNR {snr}: {len(res)} AUCs in {time.time() - t0:.0f} s')
        if snr not in TYPE_SNR:
            continue
        normal = Y.sum(1) == 0
        jobs, keys = [], []
        for j in range(len(st_idx)):
            single = (Y[:, j] == 1) & (Y.sum(1) == 1)
            for t in sorted(set(T[single].tolist())):
                m = normal | (single & (T == t))
                for i in range(len(names)):
                    jobs.append(delayed(cv_auc)(X[m, i, :], Y[m, j]))
                    keys.append((j, t, i, int((single & (T == t)).sum())))
        t0 = time.time()
        res = Parallel(n_jobs=N_JOBS)(jobs)
        for (j, t, i, npos), auc in zip(keys, res):
            rows_type.append(dict(feeder=bus, station=data['evcs_names'][j], station_bus=st_names[j],
                                  bus=names[i], attack_type=int(t), n_attacked=npos, snr_db=snr, auc=auc))
        log(f'  {bus} part 1, by attack type, SNR {snr}: {len(res)} AUCs in {time.time() - t0:.0f} s')
    return pd.DataFrame(rows_all), pd.DataFrame(rows_type)


def part2(bus, data, dist):
    """Per-bus voltage response to the station's own power deviation in single-station attack graphs."""
    X, P, Y = data['all_x'], data['all_x_P'], data['all_y']
    names, st_idx = data['bus_names'], data['evcs_node_indices']
    normal = Y.sum(1) == 0
    v_ref, p_ref = X[normal].mean(0), P[normal].mean(0)          # mean normal day, per bus and step
    rows = []
    for j, sj in enumerate(st_idx):
        single = (Y[:, j] == 1) & (Y.sum(1) == 1)
        dp = (P[single, sj, :] - p_ref[sj]).ravel()                # station's power deviation (kW)
        dv = X[single] - v_ref                                      # (graphs, buses, steps), p.u.
        beta = np.array([(dv[:, i, :].ravel() * dp).sum() / (dp ** 2).sum() for i in range(len(names))])
        for i, b in enumerate(names):
            rows.append(dict(feeder=bus, station=data['evcs_names'][j], station_bus=names[sj], bus=b,
                             beta=beta[i], measured_share=beta[i] / beta[sj]))
    out = pd.DataFrame(rows).merge(dist, on=['station_bus', 'bus'], how='left')
    out['beta_pred'] = -out.r_eff_pu / 1000.0         # p.u. per kW with S_base = 1 MVA
    return out


def main():
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    log(f'E9 parts 1 and 2 started {time.strftime("%Y-%m-%d %H:%M:%S")}, {N_JOBS} workers')
    D, A, AT, L = [], [], [], []
    for bus in ('34bus', '123bus'):
        data = load(bus)
        log(f'{bus}: {data["all_x"].shape[0]} graphs, {data["all_x"].shape[1]} buses, '
            f'stations {[data["evcs_map"][e] for e in data["evcs_names"]]}')
        G = per_unit_tree(bus)
        st_names = [data['evcs_map'][e] for e in data['evcs_names']]
        dist = footprint(G, data['bus_names'], st_names, ROOT_BUS[bus])
        dist.insert(0, 'feeder', bus)
        D.append(dist)
        L.append(part2(bus, data, dist.drop(columns='feeder')))
        a, at = part1(bus, data, log)
        A.append(a); AT.append(at)
        del data
    D, A, AT, L = (pd.concat(x, ignore_index=True) for x in (D, A, AT, L))
    for df, name in ((D, 'e9_distance.csv'), (A, 'e9_auc_all.csv'), (AT, 'e9_auc_type.csv'),
                     (L, 'e9_lindistflow.csv')):
        df.to_csv(OUT / name, index=False)
    # Short summary for the plan
    m = A.merge(D, on=['feeder', 'station_bus', 'bus'])
    for bus in ('34bus', '123bus'):
        for snr in SNR_DB:
            s = m[(m.feeder == bus) & (m.snr_db.isna() if snr is None else m.snr_db == snr)]
            near, far = s[s.d_elec <= 0.1].auc.mean(), s[s.d_elec >= 0.9].auc.mean()
            log(f'  {bus} SNR {snr}: mean AUC, d <= 0.1: {near:.3f}; d >= 0.9: {far:.3f}; '
                f'Spearman(AUC, d) = {s.auc.corr(s.d_elec, method="spearman"):.2f}')
        l = L[(L.feeder == bus) & (L.regulator_category != 'shielded by a regulator')]
        r = L[(L.feeder == bus) & (L.regulator_category == 'shielded by a regulator')]
        slope = (l.beta * l.beta_pred).sum() / (l.beta_pred ** 2).sum()
        log(f'  {bus} LinDistFlow, unshielded pairs (n = {len(l)}): corr(measured, predicted beta) = '
            f'{l.beta.corr(l.beta_pred):.2f}, slope {slope:.2f}; without the regulators corr = '
            f'{l.beta.corr(-l.r_root_pu / 1000):.2f}; shielded pairs (n = {len(r)}): mean |beta| '
            f'{r.beta.abs().mean():.2e} against {l.beta.abs().mean():.2e} unshielded')
    log(f'finished {time.strftime("%Y-%m-%d %H:%M:%S")}')
    (OUT / 'compute_e9.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
