"""E10a reference: the leakage floor for an attack type that was seen in training.

A held-out run (e10_holdout_h<t>) scores FR on the normal test graphs plus every graph of the held-out
type. Its reference is FR of E3, which saw every type in training, scored on the same kind of test set:
the normal graphs and the graphs of that type in the E3 test split. This script takes the saved E3 test
logits with the withdrawn power removed (test_occ) and the test indices and recomputes the
forgotten-station ROC-AUC on that subset, per station as src.training.evaluate_model does. On the whole
test split it must give the ROC_forget of the E3 rows, which the output checks (Check_diff).
Only reads the E3 outputs and the data cache; writes e10_seen_floor_<bus>.csv to results/2026-10-15_E4E10
or to --out.

Usage:
    python scripts/e10_seen_floor.py --bus 34bus --backbone GCN --seeds 42 43 44 45 46 --scenarios S1 S2 S3
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]             # the repository
RUN = ROOT / 'results' / '2026-10-15_E4E10'
E3 = RUN.parent / '2026-10-01_E3'
sys.path.insert(0, str(ROOT))

from scripts.train import load_experiment, load_data, resolve_feature_mode  # noqa: E402


def forgotten_roc(y, logits, forget):
    """Mean ROC-AUC over the forgotten stations, 0.5 for a station with one class (as evaluate_model)."""
    out = []
    for c in forget:
        out.append(roc_auc_score(y[:, c], logits[:, c]) if len(np.unique(y[:, c])) > 1 else 0.5)
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--bus', required=True)
    p.add_argument('--backbone', default='GCN')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--scenarios', nargs='+', required=True)
    p.add_argument('--data-cache', default=str(RUN / 'data_cache'))
    p.add_argument('--data-dir', default=str(ROOT.parent / 'Source' / 'PB_data' / '3_EVCS Attacks'),
                   help='PowerBench EVCS data, as --data_dir of train.py')
    p.add_argument('--out', default=str(RUN), help='output folder')
    args = p.parse_args()

    exp = load_experiment(args.bus, Path(args.data_dir))
    data = load_data(exp, args.bus, resolve_feature_mode(None, exp), args.data_cache)
    y_all, t_all = np.asarray(data['all_y']), np.asarray(data['all_attack_type'])
    rows = [pd.read_csv(f) for d in ('main', 'quiet', 'rest') for f in glob.glob(str(E3 / f'{d}_{args.bus}' / '*_raw.csv'))]
    e3 = pd.concat(rows, ignore_index=True)
    e3 = e3[(e3.Method == 'Retrain') & (e3.Backbone == args.backbone)].drop_duplicates(['Scenario', 'Seed'], keep='last')
    e3 = e3.set_index(['Scenario', 'Seed']).ROC_forget
    out = []
    for scen in args.scenarios:
        forget = list(range(int(scen[1:])))
        for seed in args.seeds:
            files = [f for d in ('main', 'quiet', 'rest')
                     for f in glob.glob(str(E3 / f'{d}_{args.bus}' / 'outputs' / f'{args.backbone}_{scen}_{seed}.pt'))]
            if not files:
                print(f'  {args.backbone} {scen} s{seed}: no saved outputs, skipped')
                continue
            d = torch.load(files[-1], map_location='cpu', weights_only=False)
            logits = np.asarray(d['logits']['Retrain']['test_occ'])
            idx = np.asarray(d['idx_test'])
            y, t = y_all[idx], t_all[idx]
            full = forgotten_roc(y, logits, forget)
            check = full - e3.get((scen, seed), np.nan)
            out.append(dict(feeder=args.bus, Backbone=args.backbone, Scenario=scen, Seed=seed, type='all',
                            n_graphs=len(idx), ROC_forget=full, Check_diff=check))
            for k in (1, 2, 3, 4):
                sub = np.isin(t, [0, k])
                out.append(dict(feeder=args.bus, Backbone=args.backbone, Scenario=scen, Seed=seed, type=str(k),
                                n_graphs=int(sub.sum()), ROC_forget=forgotten_roc(y[sub], logits[sub], forget)))
            print(f'  {args.backbone} {scen} s{seed}: whole test split {full:.4f}, E3 row differs by {check:+.1e}')
    df = pd.DataFrame(out)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / f'e10_seen_floor_{args.bus}.csv', index=False)
    print(df.groupby(['Scenario', 'type']).ROC_forget.mean().round(3).unstack())
    print(f'max |Check_diff| {df.Check_diff.abs().max():.1e}; {len(df)} rows written')


if __name__ == '__main__':
    main()
