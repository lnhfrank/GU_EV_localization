"""E9: per-station leakage from all bus voltages together, next to the GNN floor.

The GNN floor of a station is the ROC-AUC of full retraining (FR) on that station in the full-withdrawal
scenario (34-bus S3, 123-bus S5), where the model sees the phase-averaged voltages of all buses and no
charging power. Here a logistic regression gets the same input, the voltages of all buses (buses x 288
five-minute steps, standardized), and the same task, the station's attack label on all graphs. It uses
the splits of the GNN runs (src.data.stratified_split_multilabel with the run seed; fit on the 70%
training part, scored on the 15% test part) and their noise (src.noise.add_voltage_noise with the run
seed): seeds 42 to 51 without noise as in E3, seeds 42 to 46 at 20, 10 and 0 dB as in E9 part 4.
GNN floors: E3 (Retrain rows; 34-bus GCN S3 s47 from redo_cap_34bus) and results/noise.
Writes e9_joint_auc.csv (one row per feeder, noise level, seed and station), e9_joint_vs_floor.csv (means)
and compute_e9_joint.txt to results/leakage, or to the folder given as the first argument.
"""
import glob
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]             # the repository
sys.path.insert(0, str(ROOT))
from src.data import stratified_split_multilabel  # noqa: E402
from src.noise import add_voltage_noise  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'leakage'   # the output folder
OUT.mkdir(parents=True, exist_ok=True)
E3, E9N = ROOT / 'results' / 'comparison', ROOT / 'results' / 'noise'
FULL = {'34bus': 'S3', '123bus': 'S5'}
RUNS = [(None, s) for s in range(42, 52)] + [(snr, s) for snr in (20, 10, 0) for s in range(42, 47)]
N_JOBS = 25


def load_cache(bus):
    with open(sorted(glob.glob(str(E3 / 'data_cache' / f'{bus}_raw_*.pkl')))[0], 'rb') as f:
        return pickle.load(f)


def joint_aucs(bus, snr, seed):
    """Test-set ROC-AUC of every station for one run seed and noise level."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    data = load_cache(bus)
    cfg = yaml.safe_load(open(ROOT / 'config' / f'{bus}.yaml'))['training']
    x = data['all_x'] if snr is None else add_voltage_noise(data['all_x'], snr, seed)[0]
    X = x.reshape(len(x), -1)
    Y = data['all_y']
    tr, _, te = stratified_split_multilabel(Y, cfg['test_size'], cfg['val_ratio'], seed)
    rows = []
    for j, name in enumerate(data['evcs_names']):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced',
                                                                 solver='lbfgs', C=1.0))
        clf.fit(X[tr], Y[tr, j])
        rows.append(dict(feeder=bus, snr_db=snr, seed=seed, station=name,
                         station_bus=data['evcs_map'][name],
                         auc=roc_auc_score(Y[te, j], clf.predict_proba(X[te])[:, 1])))
    return rows


def gnn_floors(bus):
    """Per-trial FR ROC-AUC of every station in the full-withdrawal scenario, for each noise level."""
    def read(paths):
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        return df[(df.Method == 'Retrain') & (df.Scenario == FULL[bus])]
    clean = read([p for d in ('main', 'quiet', 'rest') for p in glob.glob(str(E3 / f'{d}_{bus}' / '*_raw.csv'))])
    if bus == '34bus' and (E3 / 'redo_cap_34bus').exists():   # the Retrain row that hit the old cap
        redo = read(glob.glob(str(E3 / 'redo_cap_34bus' / '*_raw.csv')))
        key = ['Backbone', 'Seed']
        clean = pd.concat([clean.merge(redo[key], on=key, how='left', indicator=True)
                           .query('_merge == "left_only"').drop(columns='_merge'), redo])
    out = [clean.assign(snr_db=np.nan)]
    for snr in (20, 10, 0):
        out.append(read(glob.glob(str(E9N / f'{bus}_snr{snr}' / '*_raw.csv'))).assign(snr_db=float(snr)))
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    lines = [f'E9 joint voltage-only AUC started {time.strftime("%Y-%m-%d %H:%M:%S")}, {N_JOBS} workers']
    jobs = [(bus, snr, seed) for bus in FULL for snr, seed in RUNS]
    res = Parallel(n_jobs=N_JOBS)(delayed(joint_aucs)(*j) for j in jobs)
    auc = pd.DataFrame([r for rows in res for r in rows])
    auc.to_csv(OUT / 'e9_joint_auc.csv', index=False)
    summary = []
    for bus in FULL:
        fl = gnn_floors(bus)
        a = auc[auc.feeder == bus]
        for snr in (None, 20, 10, 0):
            sa = a[a.snr_db.isna()] if snr is None else a[a.snr_db == snr]
            sf = fl[fl.snr_db.isna()] if snr is None else fl[fl.snr_db == snr]
            for k, (name, g) in enumerate(sa.groupby('station', sort=False)):
                col = f'ROC_EVCS{int(name.split()[-1])}'
                summary.append(dict(feeder=bus, snr_db=snr, station=name, station_bus=g.station_bus.iloc[0],
                                    auc_joint=g.auc.mean(), auc_joint_sd=g.auc.std(), n_seeds=len(g),
                                    floor_gnn=sf[col].mean(), floor_gnn_sd=sf[col].std(), n_trials=len(sf)))
    s = pd.DataFrame(summary)
    s.to_csv(OUT / 'e9_joint_vs_floor.csv', index=False)
    for bus in FULL:
        for snr in (None, 20, 10, 0):
            g = s[(s.feeder == bus) & (s.snr_db.isna() if snr is None else s.snr_db == snr)]
            pairs = ', '.join(f'{r.station} {r.auc_joint:.3f} / {r.floor_gnn:.3f}' for r in g.itertuples())
            lines.append(f'{bus} {"clean" if snr is None else f"{snr} dB"}: joint logistic / GNN floor per station: '
                         f'{pairs}; mean {g.auc_joint.mean():.3f} / {g.floor_gnn.mean():.3f}')
    lines.append(f'finished in {time.time() - t0:.0f} s')
    (OUT / 'compute_e9_joint.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
