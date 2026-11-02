"""Figure B: the forgotten stations' ROC-AUC of every method next to full retraining (FR), and the floor
under measurement noise.

Methods: all E3 trials (3 backbones x 10 seeds, every scenario), the rows of
results/2026-10-12_E3ft/e3ft_budget_trials.csv (Original with the power removed, GIF, IDEA, FT 10 to 100
epochs and to convergence, Retrain with the redo_cap replacement, Retrain_b) plus the Original model with
the power kept, from the E3 raw results. Noise: E9 part 4 (results/2026-10-06_E9), full withdrawal
(34-bus S3, 123-bus S5), seeds 42 to 46, without noise and at 20, 10 and 0 dB.
Writes figB_methods_trials.csv and figB_noise_trials.csv (one row per trial and method) to
results/2026-10-12_figB, or to the folder given as the first argument.
"""
import glob
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / 'results'
HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else R / '2026-10-12_figB'   # the output folder
HERE.mkdir(parents=True, exist_ok=True)
E3, E3FT, E9N = R / '2026-10-01_E3', R / '2026-10-12_E3ft', R / '2026-10-06_E9'
FEEDERS = ('34bus', '123bus')
NOISE_NAME = {'Finetune_pure': 'FT_e25', 'FTpure_e50': 'FT_e50', 'FTpure_e100': 'FT_e100'}
KEY = ['feeder', 'Scenario', 'Backbone', 'Seed', 'method']


def raw(paths):
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main():
    t = pd.read_csv(E3FT / 'e3ft_budget_trials.csv')
    t = t[t.method.isin(['Mask_only', 'GIF', 'IDEA_s0.001', 'IDEA_cert', 'FT_e10', 'FT_e25', 'FT_e50', 'FT_e100',
                         'WarmStart', 'Retrain', 'Retrain_b'])]
    orig = []
    for bus in FEEDERS:
        o = raw([p for d in ('main', 'quiet', 'rest') for p in glob.glob(str(E3 / f'{d}_{bus}' / '*_raw.csv'))])
        o = o[o.Method == 'Original'].assign(feeder=bus, method='Original_power')
        orig.append(o)
    m = pd.concat([t[KEY + ['ROC_forget']], pd.concat(orig)[KEY + ['ROC_forget']]], ignore_index=True)
    assert not m.duplicated(KEY).any()
    m.to_csv(HERE / 'figB_methods_trials.csv', index=False)

    rows = []
    for bus in FEEDERS:
        for level, snr in (('clean', None), ('snr20', 20), ('snr10', 10), ('snr0', 0)):
            n = raw(glob.glob(str(E9N / f'{bus}_{level}' / '*_raw.csv')))
            n = n.assign(feeder=bus, snr_db=snr, method=n.Method.map(lambda x: NOISE_NAME.get(x, x)))
            n.loc[n.method == 'Original', 'method'] = 'Original_power'
            rows.append(n[['feeder', 'snr_db', 'Scenario', 'Backbone', 'Seed', 'method', 'ROC_forget']])
    noise = pd.concat(rows, ignore_index=True)
    assert not noise.duplicated(['feeder', 'snr_db', 'Backbone', 'Seed', 'method']).any()
    noise.to_csv(HERE / 'figB_noise_trials.csv', index=False)
    print(m.groupby(['feeder', 'Scenario', 'method']).ROC_forget.agg(['size', 'mean']).round(3).to_string())
    print(noise.groupby(['feeder', 'snr_db', 'method'], dropna=False).ROC_forget.agg(['size', 'mean']).round(3)
          .to_string())


if __name__ == '__main__':
    main()
