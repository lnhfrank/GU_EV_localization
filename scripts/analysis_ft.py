"""Fine-tuning budgets for the main table and the budget figures.

Per trial (feeder, scenario, backbone, seed) and method, the test-set metrics and their difference from
full retraining (FR, Retrain) of the same trial. Sources:
  E3 (results/2026-10-01_E3): Mask_only, GIF, IDEA_s0.001, IDEA_cert, FT 25 epochs (Finetune_pure),
      WarmStart (fine-tuning to convergence with the training procedure), Retrain, Retrain_b. The Retrain
      and Retrain_b rows of 34-bus GCN S3 s47 and GIN S3 s49 come from redo_cap_34bus (the 600-epoch cap).
  E3ft (this folder): FT 10, 50 and 100 epochs (FTpure_e10, _e50, _e100) on every E3 trial.
  E5 (results/2026-10-02_E5/budget_<bus>): FT 5, 200 and 300 epochs, S1 and seeds 42 to 46 only.
Timing: quiet phase only (one process per GPU), S1, seeds 42 to 44: FT from this folder's quiet_<bus>,
the other methods from E3's quiet_<bus>; for 123-bus GIN s42 and GCN s44 E3 took the times from
quiet_redo_123bus, so this script does the same.
Writes e3ft_budget_trials.csv, e3ft_timing.csv and analysis_ft.txt.
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1] / 'results' / '2026-10-12_E3ft'   # the E3ft runs
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN   # a test run writes elsewhere
OUT.mkdir(parents=True, exist_ok=True)
R = RUN.parent
E3, E5 = R / '2026-10-01_E3', R / '2026-10-02_E5'
FEEDERS = ('34bus', '123bus')
FULL = {'34bus': 'S3', '123bus': 'S5'}           # all stations withdrawn
METRICS = ['Macro_ROC', 'ROC_forget', 'MIA_forget', 'Macro_F1']
TIMES = ['Time', 'T_grad', 'T_step', 'T_bn', 'T_ft', 'T_val', 'Epochs_run']
NAME = {'Finetune_pure': 'FT_e25', 'FTpure_e5': 'FT_e5', 'FTpure_e10': 'FT_e10', 'FTpure_e50': 'FT_e50',
        'FTpure_e100': 'FT_e100', 'FTpure_e200': 'FT_e200', 'FTpure_e300': 'FT_e300'}
E3_METHODS = ['Mask_only', 'GIF', 'IDEA_s0.001', 'IDEA_cert', 'Finetune_pure', 'WarmStart', 'Retrain', 'Retrain_b']


def rows(folders, source):
    out = []
    for folder in folders:
        for f in sorted(glob.glob(str(folder / '*_raw.csv'))):
            for r in csv.DictReader(open(f)):
                r['source'] = source
                out.append(r)
    return out


def frame(rs):
    df = pd.DataFrame(rs)
    if df.empty:
        return pd.DataFrame(columns=['Backbone', 'Scenario', 'Seed', 'Method', 'source'] + METRICS + TIMES)
    for c in METRICS + TIMES:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Seed'] = df['Seed'].astype(int)
    return df


def main():
    out = []
    lines = []
    def log(s):
        print(s); lines.append(s)
    tables, timing = [], []
    for bus in FEEDERS:
        e3 = frame(rows([E3 / f'{d}_{bus}' for d in ('main', 'quiet', 'rest')], 'E3'))
        e3 = e3[e3.Method.isin(E3_METHODS)]
        if bus == '34bus':                          # replace the two Retrain rows that hit the old cap
            redo = frame(rows([E3 / 'redo_cap_34bus'], 'E3 redo_cap'))
            redo = redo[redo.Method.isin(['Retrain', 'Retrain_b'])]
            key = ['Backbone', 'Scenario', 'Seed', 'Method']
            e3 = pd.concat([e3.merge(redo[key], on=key, how='left', indicator=True)
                            .query('_merge == "left_only"').drop(columns='_merge'), redo])
        ft = frame(rows([RUN / f'main_{bus}'], 'E3ft'))
        ft = ft[ft.Method.isin(['FTpure_e10', 'FTpure_e50', 'FTpure_e100'])]
        e5 = frame(rows([E5 / f'budget_{bus}'], 'E5'))
        e5 = e5[e5.Method.isin(['FTpure_e5', 'FTpure_e200', 'FTpure_e300'])]
        df = pd.concat([e3, ft, e5], ignore_index=True)
        df['method'] = df.Method.map(lambda m: NAME.get(m, m))
        dup = df.duplicated(['Backbone', 'Scenario', 'Seed', 'method'])
        assert not dup.any(), df[dup][['Backbone', 'Scenario', 'Seed', 'method']].head()
        fr = df[df.method == 'Retrain'].set_index(['Backbone', 'Scenario', 'Seed'])[METRICS]
        df = df.join(fr, on=['Backbone', 'Scenario', 'Seed'], rsuffix='_FR')
        for c in METRICS:
            df[f'gap_{c}'] = df[c] - df[f'{c}_FR']
        df.insert(0, 'feeder', bus)
        tables.append(df[['feeder', 'Scenario', 'Backbone', 'Seed', 'method', 'source'] + METRICS
                         + [f'gap_{c}' for c in METRICS] + ['Epochs_run']])
        # Quiet timing
        q_ft = frame(rows([RUN / f'quiet_{bus}'], 'E3ft quiet'))
        q_e3 = frame(rows([E3 / f'quiet_{bus}'], 'E3 quiet'))
        q_e3 = q_e3[q_e3.Method.isin(E3_METHODS)]
        q_e3.loc[q_e3.Method == 'Finetune_pure', 'Method'] = 'FT_e25_E3quiet'   # the 10/1 session, a check
        if bus == '123bus':
            redo = frame(rows([E3 / 'quiet_redo_123bus'], 'E3 quiet_redo'))
            key = ['Backbone', 'Seed', 'Method']
            q_e3 = pd.concat([q_e3.merge(redo[key], on=key, how='left', indicator=True)
                              .query('_merge == "left_only"').drop(columns='_merge'),
                              redo[redo.Method.isin(E3_METHODS) & (redo.Method != 'Finetune_pure')]])
        q_e3.loc[q_e3.Method == 'Finetune_pure', 'Method'] = 'FT_e25_E3quiet'
        q = pd.concat([q_ft, q_e3], ignore_index=True)
        q = q[q.Scenario == 'S1']
        q['method'] = q.Method.map(lambda m: NAME.get(m, m))
        q.insert(0, 'feeder', bus)
        timing.append(q[['feeder', 'Scenario', 'Backbone', 'Seed', 'method', 'source'] + [c for c in TIMES if c in q]])
    T = pd.concat(tables, ignore_index=True)
    Q = pd.concat(timing, ignore_index=True)
    T.to_csv(OUT / 'e3ft_budget_trials.csv', index=False)
    Q.to_csv(OUT / 'e3ft_timing.csv', index=False)

    order = ['Mask_only', 'GIF', 'IDEA_s0.001', 'IDEA_cert', 'FT_e5', 'FT_e10', 'FT_e25', 'FT_e50', 'FT_e100',
             'FT_e200', 'FT_e300', 'WarmStart', 'Retrain_b']
    for bus in FEEDERS:
        for scen in ('S1', FULL[bus]):
            s = T[(T.feeder == bus) & (T.Scenario == scen)]
            log(f'== {bus} {scen}: |forgotten-station ROC - FR| and |Macro ROC - FR|, mean over trials (n)')
            for m in order:
                g = s[s.method == m]
                if len(g):
                    log(f'   {m:12s} n={len(g):3d}  ROC_forget {g.gap_ROC_forget.abs().mean():.4f} '
                        f'(signed {g.gap_ROC_forget.mean():+.4f})  Macro_ROC {g.gap_Macro_ROC.abs().mean():.4f}')
        t = Q[Q.feeder == bus].groupby('method').Time.median()
        log(f'== {bus} quiet timing, S1, median seconds per request: ' +
            ', '.join(f'{m} {t[m]:.1f}' for m in order + ['FT_e25_E3quiet', 'Retrain'] if m in t))
    # Same-trial check: E3ft and E5 both have FT 10, 50, 100 at S1 seeds 42-46 (GPU run-to-run noise only)
    for bus in FEEDERS:
        e5 = frame(rows([E5 / f'budget_{bus}'], 'E5'))
        a = T[(T.feeder == bus) & (T.Scenario == 'S1') & T.method.isin(['FT_e10', 'FT_e50', 'FT_e100'])
              & T.Seed.between(42, 46)]
        b = e5[e5.Method.isin(['FTpure_e10', 'FTpure_e50', 'FTpure_e100'])].copy()
        b['method'] = b.Method.map(NAME)
        m = a.merge(b, left_on=['Backbone', 'Seed', 'method'], right_on=['Backbone', 'Seed', 'method'])
        log(f'== {bus} E3ft against E5 on the same S1 trials: max |ROC_forget difference| '
            f'{(m.ROC_forget_x - m.ROC_forget_y).abs().max():.4f}, mean {(m.ROC_forget_x - m.ROC_forget_y).abs().mean():.4f} (n = {len(m)})')
    (OUT / 'analysis_ft.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
