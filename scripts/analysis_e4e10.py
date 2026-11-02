"""E4, the E5 from-scratch control and E10.

Every gap is a method's test metric minus that of full retraining (FR, Retrain) in the same trial; the
reference band is Retrain_b, the retraining with seed + 1000 on the same split. Forgotten-station
Fidelity and probability difference come from the saved per-graph test logits with the withdrawn power
removed (test_occ), as in results/2026-10-15_8500/analysis_8500.py.
  E4: sequential requests (this folder, e4_<bus>), step j = scenario S_j, every method starting from its
      model after S_{j-1}; compared with the same requests served from the original model in E3 (FT 25,
      GIF+FT, Mask_only) and E3ft (FT 100), seeds 42 to 46.
  E5 control: fine-tuning from a freshly initialized model (scratch_<bus>, S1), against fine-tuning from
      the original model with the same learning rate and epochs (results/2026-10-02_E5/budget_<bus>);
      FR comes from the scratch runs, which read the E3 Retrain from the cache.
  E10a: one attack type held out of training and validation (e10_holdout_h<t>, 34-bus GCN, S1 and S3; S2 in
      results/2026-10-19_E10a),
      against the same type seen in training: FR of E3 on the normal and that type's graphs of its test
      split (e10_seen_floor.py, e10_seen_floor_34bus.csv); on the 123-bus feeder, S1 to S5 with FR and
      Retrain_b only (results/2026-10-17_E10a, e10_seen_floor_123bus.csv), once those trials are there.
  E10b: wrong lines at test time (e10_topology/topology.csv, scripts/eval_topology.py), summarized here.
  E10c: the withdrawn power zeroed with a mask channel (e10_mask), against zeroing after
      standardization in E3 (34-bus GCN, seeds 42 to 51, S1 to S3).
  8500: the same control at S7 (scratch_8500, 50 and 100 epochs) against FT 50 and 100 epochs of the
      8500 runs (results/2026-10-15_8500), once those trials are there.
The utility in E4 is the ROC-AUC of the retained stations (ROC_retain); Macro ROC averages the forgotten
stations in as well, so its gap grows with their number alone.
Writes e4e10_trials.csv, analysis_e4e10.txt and, for the figures, e4_steps.csv, e4_matched.csv, e5_control.csv,
e10_holdout.csv and e10_mask.csv to results/2026-10-15_E4E10, or to the folder given as the first argument.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R = Path(__file__).resolve().parents[1] / 'results'
RUN = R / '2026-10-15_E4E10'          # the E4 and E10 runs
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN   # the output folder
E3, E3FT, E5 = R / '2026-10-01_E3', R / '2026-10-12_E3ft', R / '2026-10-02_E5'
NAME = {'Finetune_pure': 'FT_e25', 'FTpure_e100': 'FT_e100', 'GIF_FT': 'GIF+FT', 'FTpure_e200': 'FT_e200',
        'FTpure_e300': 'FT_e300', 'FTpure_lr1e-3_e100': 'FT_lr1e-3_e100', 'FTpure_lr1e-3_e300': 'FT_lr1e-3_e300',
        'Scratch_e100': 'Scratch_e100', 'Scratch_e200': 'Scratch_e200', 'Scratch_e300': 'Scratch_e300',
        'Scratch_lr1e-3_e100': 'Scratch_lr1e-3_e100', 'Scratch_lr1e-3_e300': 'Scratch_lr1e-3_e300',
        'FTpure_e50': 'FT_e50', 'Scratch_e50': 'Scratch_e50', 'FTpure_e10': 'FT_e10', 'FTpure_e75': 'FT_e75',
        'FTpure_e125': 'FT_e125'}
# (start, learning rate, epochs) of the fine-tuning arms compared in the E5 control
BUDGET = {'Scratch_e50': ('scratch', 1e-4, 50), 'Scratch_e100': ('scratch', 1e-4, 100),
          'Scratch_e200': ('scratch', 1e-4, 200), 'Scratch_e300': ('scratch', 1e-4, 300),
          'Scratch_lr1e-3_e100': ('scratch', 1e-3, 100), 'Scratch_lr1e-3_e300': ('scratch', 1e-3, 300),
          'FT_e50': ('original', 1e-4, 50), 'FT_e100': ('original', 1e-4, 100), 'FT_e200': ('original', 1e-4, 200),
          'FT_e300': ('original', 1e-4, 300), 'FT_lr1e-3_e100': ('original', 1e-3, 100),
          'FT_lr1e-3_e300': ('original', 1e-3, 300)}
E8500 = R / '2026-10-15_8500'
E10A = R / '2026-10-17_E10a'
E10A34 = R / '2026-10-19_E10a'
E4B = R / '2026-10-19_E4b'
KEY = ['Backbone', 'Scenario', 'Seed']


def rows(folders, **tags):
    out = [pd.read_csv(f) for d in folders for f in sorted(glob.glob(str(d / '*_raw.csv')))]
    df = pd.concat(out, ignore_index=True)
    df['method'] = df.Method.map(lambda m: NAME.get(m, m))
    for k, v in tags.items():
        df[k] = v
    return df


def e3_rows(bus):
    df = rows([E3 / f'{d}_{bus}' for d in ('main', 'quiet', 'rest')])
    if bus == '34bus':                          # the two rows rerun after the 600-epoch cap
        redo = rows([E3 / 'redo_cap_34bus'])
        df = pd.concat([df.merge(redo[KEY + ['Method']], on=KEY + ['Method'], how='left', indicator=True)
                        .query('_merge == "left_only"').drop(columns='_merge'), redo], ignore_index=True)
    ft = rows([E3FT / f'main_{bus}'])
    ft = ft[ft.Method.isin(['FTpure_e10', 'FTpure_e50', 'FTpure_e100'])]
    return pd.concat([df, ft], ignore_index=True).drop_duplicates(KEY + ['Method'], keep='last')


def with_gaps(df, fr=None):
    """Signed gap of every row to FR of the same trial (FR from df itself unless given)."""
    fr = (df if fr is None else fr)
    cols = [c for c in ('ROC_forget', 'ROC_retain', 'Macro_ROC', 'MIA_forget') if c in df and c in fr]
    fr = fr[fr.Method == 'Retrain'].drop_duplicates(KEY).set_index(KEY)[cols]
    df = df.join(fr, on=KEY, rsuffix='_FR')
    for c in cols:
        df[f'gap_{c}'] = df[c] - df[f'{c}_FR']
    return df


def agreement(folder, tag=''):
    """Forgotten-station Fidelity and probability difference to FR from the saved test logits."""
    out = []
    for f in sorted(glob.glob(str(folder / 'outputs' / '*.pt'))):
        parts = Path(f).stem.split('_')
        bb, scen, seed = parts[0], parts[1], int(parts[-1])
        d = torch.load(f, map_location='cpu', weights_only=False)
        forget = list(range(int(scen[1:])))
        p = {m: torch.sigmoid(torch.as_tensor(v['test_occ'])).numpy()[:, forget] for m, v in d['logits'].items()}
        for m, pm in p.items():
            out.append(dict(Backbone=bb, Scenario=scen, Seed=seed, method=NAME.get(m, m), tag=tag,
                            fidelity=float(((pm > 0.5) == (p['Retrain'] > 0.5)).mean()),
                            probdiff=float(np.abs(pm - p['Retrain']).mean())))
    return pd.DataFrame(out)


def main():
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    keep, steps, control, holdout, masks, matched = [], [], [], [], [], []
    # ── E4 ──
    for bus, K in (('34bus', 3), ('123bus', 5)):
        seq = with_gaps(rows([RUN / f'e4_{bus}'], feeder=bus, exp='E4'))
        seq = seq.merge(agreement(RUN / f'e4_{bus}'), on=KEY + ['method'], how='left')
        one = e3_rows(bus)
        if glob.glob(str(E4B / f'main_{bus}' / '*_raw.csv')):  # one-shot FT 75 and 125 (E4b)
            extra = rows([E4B / f'main_{bus}'])
            one = pd.concat([one, extra[extra.method.isin(['FT_e75', 'FT_e125'])]], ignore_index=True)
        one = with_gaps(one)
        one = one[one.Seed.between(42, 46)]
        keep.append(seq)
        for mode, d in (('sequential', seq), ('one-shot', one)):
            d = d[d.method.isin(['Mask_only', 'FT_e25', 'FT_e100', 'GIF+FT', 'Retrain', 'Retrain_b'])]
            steps.append(d.assign(feeder=bus, mode=mode, step=d.Scenario.str[1:].astype(int))[
                ['feeder', 'mode', 'method', 'step', 'Backbone', 'Seed', 'ROC_forget', 'ROC_retain', 'Macro_ROC',
                 'MIA_forget', 'gap_ROC_forget', 'gap_ROC_retain', 'gap_Macro_ROC', 'gap_MIA_forget']
                + (['fidelity'] if mode == 'sequential' else [])])
        log(f'== E4 {bus}: gap to FR, sequential (one-shot from E3), mean over 15 chains: forgotten-station ROC; '
            f'|gap|; Fidelity F; retained-station ROC R, the utility (no retained station at S{K}); '
            f'forgotten-station MIA. Bands: mean |Retrain_b - Retrain| of the forgotten-station ROC and MIA')
        for j in range(1, K + 1):
            s = seq[seq.Scenario == f'S{j}']
            o = one[one.Scenario == f'S{j}']
            b = s[s.method == 'Retrain_b']
            parts = []
            for m in ('Mask_only', 'FT_e25', 'FT_e100', 'GIF+FT'):
                g, h = s[s.method == m], o[o.method == m]
                parts.append(f'{m} {g.gap_ROC_forget.mean():+.3f} ({h.gap_ROC_forget.mean():+.3f}) '
                             f'|{g.gap_ROC_forget.abs().mean():.3f}| F {g.fidelity.mean():.3f} '
                             f'R {g.gap_ROC_retain.mean():+.4f} ({h.gap_ROC_retain.mean():+.4f}) '
                             f'MIA {g.gap_MIA_forget.mean():+.3f} ({h.gap_MIA_forget.mean():+.3f})')
            log(f'   S{j}: band {b.gap_ROC_forget.abs().mean():.4f}, MIA {b.gap_MIA_forget.abs().mean():.3f}; '
                + '; '.join(parts))
        f100 = seq[(seq.method == 'FT_e100') & (seq.Scenario == f'S{K}')].groupby('Backbone').gap_MIA_forget.mean()
        log(f'   FT_e100 sequential at S{K}, MIA gap by backbone: '
            + ', '.join(f'{bb} {v:+.3f}' for bb, v in f100.items()))
        # Matched total epochs: sequential FT 25 at S_j has fine-tuned 25 j epochs in total; compare it with
        # one-shot FT of that budget (E3, E3ft, E4b), or the two one-shot budgets around it
        budgets = {e: f'FT_e{e}' for e in (10, 25, 50, 75, 100, 125) if (one.method == f'FT_e{e}').any()}
        log(f'   {bus} matched total epochs, gap to FR (forgotten-station ROC / MIA): sequential FT 25 at S_j '
            f'(25 j epochs in total) against one-shot FT of the same or the nearest budgets; one-shot FT to '
            f'convergence with its mean epochs')
        for j in range(1, K + 1):
            s = seq[(seq.Scenario == f'S{j}') & (seq.method == 'FT_e25')]
            o = one[one.Scenario == f'S{j}']
            total = 25 * j
            near = [e for e in budgets if e == total] or \
                   [max(e for e in budgets if e < total)] + [min((e for e in budgets if e > total), default=None)]
            parts = []
            for e in (e for e in near if e is not None):
                g = o[o.method == budgets[e]]
                parts.append(f'one-shot FT {e} {g.gap_ROC_forget.mean():+.3f} / {g.gap_MIA_forget.mean():+.3f}')
            w = o[o.method == 'WarmStart']
            parts.append(f'to convergence ({w.Epochs_run.mean():.0f} ep) {w.gap_ROC_forget.mean():+.3f} / '
                         f'{w.gap_MIA_forget.mean():+.3f}')
            log(f'   S{j}: sequential FT 25 ({total} ep) {s.gap_ROC_forget.mean():+.3f} / '
                f'{s.gap_MIA_forget.mean():+.3f}; ' + '; '.join(parts))
            # rows for the figure: one-shot FT at every budget, sequential FT 25 at its total, the band, and FR
            cols = ['Backbone', 'Seed', 'ROC_forget', 'MIA_forget', 'gap_ROC_forget', 'gap_MIA_forget']
            for e, m in budgets.items():
                matched.append(o[o.method == m][cols].assign(feeder=bus, step=j, mode='one-shot', epochs=e))
            matched.append(s[cols].assign(feeder=bus, step=j, mode='sequential', epochs=total))
            band = seq[(seq.Scenario == f'S{j}') & (seq.method == 'Retrain_b')]
            matched.append(band[cols].assign(feeder=bus, step=j, mode='band', epochs=np.nan))
            fr = seq[(seq.Scenario == f'S{j}') & (seq.method == 'Retrain')]
            matched.append(fr[cols].assign(feeder=bus, step=j, mode='fr', epochs=np.nan))
    # ── E5 scratch control ──
    for bus in ('34bus', '123bus'):
        sc = with_gaps(rows([RUN / f'scratch_{bus}'], feeder=bus, exp='E5 scratch'))
        ft = with_gaps(rows([E5 / f'budget_{bus}']), fr=sc)
        keep.append(sc)
        for d in (sc, ft):
            d = d[d.method.isin(list(BUDGET)) | (d.method == 'Retrain_b')]
            control.append(pd.DataFrame(dict(
                feeder=bus, Scenario='S1', Backbone=d.Backbone, Seed=d.Seed, method=d.method,
                start=d.method.map(lambda m: BUDGET.get(m, ('band',))[0]),
                lr=d.method.map(lambda m: BUDGET.get(m, (0, np.nan))[1]),
                epochs=d.method.map(lambda m: BUDGET.get(m, (0, 0, np.nan))[2]),
                gap_ROC_forget=d.gap_ROC_forget, gap_Macro_ROC=d.gap_Macro_ROC)))
        log(f'== E5 control {bus}, S1: forgotten-station ROC gap to FR, from scratch vs from the original model '
            f'(same lr and epochs), mean over 15 trials; Macro ROC gap in brackets')
        for budget, a, b in (('lr 1e-4, 100 ep', 'Scratch_e100', 'FT_e100'), ('lr 1e-4, 200 ep', 'Scratch_e200', 'FT_e200'),
                             ('lr 1e-4, 300 ep', 'Scratch_e300', 'FT_e300'),
                             ('lr 1e-3, 100 ep', 'Scratch_lr1e-3_e100', 'FT_lr1e-3_e100'),
                             ('lr 1e-3, 300 ep', 'Scratch_lr1e-3_e300', 'FT_lr1e-3_e300')):
            g, h = sc[sc.method == a], ft[ft.method == b]
            log(f'   {budget}: scratch {g.gap_ROC_forget.mean():+.4f} ({g.gap_Macro_ROC.mean():+.4f}), '
                f'from theta* {h.gap_ROC_forget.mean():+.4f} ({h.gap_Macro_ROC.mean():+.4f}); '
                f'scratch above FR in {(g.gap_ROC_forget > 0).mean():.0%}, theta* in {(h.gap_ROC_forget > 0).mean():.0%}')
        band = sc[sc.method == 'Retrain_b'].gap_ROC_forget.abs().mean()
        log(f'   band {band:.4f}; FR forgotten ROC {sc[sc.method == "Retrain"].ROC_forget.mean():.3f}')
    # ── E10a: unseen attack type ──
    e3 = with_gaps(e3_rows('34bus'))
    e3g = e3[(e3.Backbone == 'GCN')]
    hcols = ['feeder', 'type', 'training', 'Scenario', 'method', 'Seed', 'ROC_forget', 'gap_ROC_forget']
    seen = pd.read_csv(RUN / 'e10_seen_floor_34bus.csv', dtype={'type': str})
    seen = seen[seen.Seed.between(42, 46)]
    holdout.append(seen[seen.type != 'all'].assign(method='Retrain', training='seen', gap_ROC_forget=np.nan)[hcols])
    log('== E10a 34-bus GCN, one attack type held out (test: normal graphs + every graph of that type), '
        'mean over seeds 42 to 46. Seen: FR of E3, which saw every type, on the normal and that type\'s '
        'graphs of its test split (e10_seen_floor.py); whole split: FR of E3 on its whole test split')
    for t in (1, 2, 3, 4):
        folders = [d for d in (RUN / f'e10_holdout_h{t}', E10A34 / f'holdout_h{t}')
                   if glob.glob(str(d / '*_raw.csv'))]     # S1 and S3 here, S2 in results/2026-10-19_E10a
        h = with_gaps(rows(folders, feeder='34bus', exp=f'E10a h{t}'))
        h = h.merge(pd.concat([agreement(d, f'h{t}') for d in folders], ignore_index=True),
                    on=KEY + ['method'], how='left')
        keep.append(h)
        holdout.append(h.assign(type=str(t), training='held out')[hcols])
        for scen in sorted(h.Scenario.unique()):
            s = h[h.Scenario == scen]
            ref = e3g[(e3g.Scenario == scen) & e3g.Seed.between(42, 46)]
            fr = s[s.method == 'Retrain'].ROC_forget.mean()
            sv = seen[(seen.Scenario == scen) & (seen.type == str(t))].ROC_forget.mean()
            log(f'   type {t} {scen}: Original {s[s.method == "Original"].ROC_forget.mean():.3f}, '
                f'Mask_only {s[s.method == "Mask_only"].ROC_forget.mean():.3f}, '
                f'FR {fr:.3f} (seen: {sv:.3f}, held out minus seen {fr - sv:+.3f}; whole split: '
                f'{ref[ref.method == "Retrain"].ROC_forget.mean():.3f}); |gap| FT 25 '
                f'{s[s.method == "FT_e25"].gap_ROC_forget.abs().mean():.3f}, FT 100 '
                f'{s[s.method == "FT_e100"].gap_ROC_forget.abs().mean():.3f}, WarmStart '
                f'{s[s.method == "WarmStart"].gap_ROC_forget.abs().mean():.3f}, band '
                f'{s[s.method == "Retrain_b"].gap_ROC_forget.abs().mean():.3f}')
    # E10a on the 123-bus feeder (results/2026-10-17_E10a): S1 to S5, FR and Retrain_b only
    if glob.glob(str(E10A / 'holdout_h*' / '*_raw.csv')):
        e3b = e3_rows('123bus')
        e3b = e3b[(e3b.Backbone == 'GCN') & e3b.Seed.between(42, 46) & (e3b.method == 'Retrain')]
        seen_b = pd.read_csv(RUN / 'e10_seen_floor_123bus.csv', dtype={'type': str})
        seen_b = seen_b[seen_b.Seed.between(42, 46)]
        holdout.append(seen_b[seen_b.type != 'all'].assign(method='Retrain', training='seen',
                                                           gap_ROC_forget=np.nan)[hcols])
        log('== E10a 123-bus GCN, the same with FR and Retrain_b only (results/2026-10-17_E10a), seeds 42 to 46')
        for t in (1, 2, 3, 4):
            folder = E10A / f'holdout_h{t}'
            if not glob.glob(str(folder / '*_raw.csv')):
                continue
            h = with_gaps(rows([folder], feeder='123bus', exp=f'E10a 123bus h{t}'))
            keep.append(h)
            holdout.append(h.assign(type=str(t), training='held out')[hcols])
            for scen in sorted(h.Scenario.unique()):
                s = h[h.Scenario == scen]
                fr = s[s.method == 'Retrain'].ROC_forget
                sv = seen_b[(seen_b.Scenario == scen) & (seen_b.type == str(t))].ROC_forget.mean()
                log(f'   type {t} {scen}: Original {s[s.method == "Original"].ROC_forget.mean():.3f}, '
                    f'Mask_only {s[s.method == "Mask_only"].ROC_forget.mean():.3f}, FR {fr.mean():.3f} '
                    f'({len(fr)} seeds; seen: {sv:.3f}, held out minus seen {fr.mean() - sv:+.3f}; whole split: '
                    f'{e3b[e3b.Scenario == scen].ROC_forget.mean():.3f}); band '
                    f'{s[s.method == "Retrain_b"].gap_ROC_forget.abs().mean():.3f}')
    # ── E10c: mask channel ──
    mk = with_gaps(rows([RUN / 'e10_mask'], feeder='34bus', exp='E10c mask'))
    mk = mk.merge(agreement(RUN / 'e10_mask', 'mask'), on=KEY + ['method'], how='left')
    keep.append(mk)
    ref = e3g[e3g.Scenario.isin(['S1', 'S2', 'S3']) & e3g.Seed.between(42, 46)]
    holdout.append(ref.assign(feeder='34bus', type='all', training='seen')[hcols])
    z = e3g[e3g.Seed.between(42, 51)]
    for name, d in (('mask', mk), ('zero_std', z)):
        masks.append(d.assign(semantics=name)[['semantics', 'Scenario', 'method', 'Seed', 'ROC_forget', 'gap_ROC_forget']])
    log('== E10c 34-bus GCN, seeds 42 to 51: mask channel vs zeroing after standardization (E3)')
    for scen in ('S1', 'S2', 'S3'):
        s, z = mk[mk.Scenario == scen], e3g[(e3g.Scenario == scen) & e3g.Seed.between(42, 51)]
        pair = s[s.method == 'Retrain'].set_index('Seed').ROC_forget - z[z.method == 'Retrain'].set_index('Seed').ROC_forget
        log(f'   {scen}: FR mask {s[s.method == "Retrain"].ROC_forget.mean():.3f} vs zero_std '
            f'{z[z.method == "Retrain"].ROC_forget.mean():.3f} (paired difference {pair.mean():+.3f}, sd {pair.std():.3f}); '
            f'Original {s[s.method == "Original"].ROC_forget.mean():.3f} vs {z[z.method == "Original"].ROC_forget.mean():.3f}; '
            f'Mask_only {s[s.method == "Mask_only"].ROC_forget.mean():.3f} vs {z[z.method == "Mask_only"].ROC_forget.mean():.3f}; '
            f'|gap| FT 25 {s[s.method == "FT_e25"].gap_ROC_forget.abs().mean():.3f} vs '
            f'{z[z.method == "FT_e25"].gap_ROC_forget.abs().mean():.3f}, FT 100 '
            f'{s[s.method == "FT_e100"].gap_ROC_forget.abs().mean():.3f} vs {z[z.method == "FT_e100"].gap_ROC_forget.abs().mean():.3f}; '
            f'band {s[s.method == "Retrain_b"].gap_ROC_forget.abs().mean():.3f} vs '
            f'{z[z.method == "Retrain_b"].gap_ROC_forget.abs().mean():.3f}')
    # ── E10b: topology errors at test time ──
    if not (RUN / 'e10_topology' / 'topology.csv').exists():   # the topology test of the letter, optional
        topo = None
    else:
        topo = pd.read_csv(RUN / 'e10_topology' / 'topology.csv')
        base = topo[topo.Perturb == 'none'].set_index(['Scenario', 'Seed', 'Method']).ROC_forget
        pt = topo[topo.Perturb != 'none'].copy()
        pt['delta'] = pt.ROC_forget.values - base.loc[list(zip(pt.Scenario, pt.Seed, pt.Method))].values
        log('== E10b 34-bus GCN, E3 models (seeds 42 to 51, S1 to S3), wrong lines at test time: change of the '
            'forgotten-station ROC against the correct topology, mean (sd; lowest single draw) over 30 trials x 5 draws')
        for kind, k in (('remove', 1), ('remove', 3), ('add', 1), ('add', 3)):
            parts = []
            for m in ('Retrain', 'Finetune_pure', 'FTpure_e100', 'WarmStart', 'Mask_only', 'Original'):
                v = pt[(pt.Perturb == kind) & (pt.K == k) & (pt.Method == m)].delta
                parts.append(f'{NAME.get(m, m)} {v.mean():+.4f} ({v.std():.3f}; {v.min():+.3f})')
            log(f'   {kind} {k}: ' + ', '.join(parts))
    # ── 8500 control at S7, once its trials are there ──
    if glob.glob(str(RUN / 'scratch_8500' / '*_raw.csv')):
        sc8 = with_gaps(rows([RUN / 'scratch_8500'], feeder='8500bus', exp='E5 scratch 8500'))
        ft8 = rows([E8500 / f'{d}_8500bus' for d in ('quiet', 'main')])
        ft8 = with_gaps(ft8[ft8.Scenario == 'S7'], fr=sc8)
        ft8 = ft8[ft8.Seed.isin(sc8.Seed.unique())]
        keep.append(sc8)
        log('== E5 control 8500bus, S7: forgotten-station ROC gap to FR, from scratch vs from the original model')
        for e in (50, 100):
            g, h = sc8[sc8.method == f'Scratch_e{e}'], ft8[ft8.method == f'FT_e{e}']
            log(f'   lr 1e-4, {e} ep: scratch {g.gap_ROC_forget.mean():+.4f} ({len(g)} trials), '
                f'from theta* {h.gap_ROC_forget.mean():+.4f} ({len(h)} trials)')
        log(f'   band {sc8[sc8.method == "Retrain_b"].gap_ROC_forget.abs().mean():.4f}')
        for d in (sc8, ft8):
            d = d[d.method.isin(list(BUDGET)) | (d.method == 'Retrain_b')]
            control.append(pd.DataFrame(dict(
                feeder='8500bus', Scenario='S7', Backbone=d.Backbone, Seed=d.Seed, method=d.method,
                start=d.method.map(lambda m: BUDGET.get(m, ('band',))[0]),
                lr=d.method.map(lambda m: BUDGET.get(m, (0, np.nan))[1]),
                epochs=d.method.map(lambda m: BUDGET.get(m, (0, 0, np.nan))[2]),
                gap_ROC_forget=d.gap_ROC_forget, gap_Macro_ROC=d.gap_Macro_ROC)))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(steps, ignore_index=True).to_csv(OUT / 'e4_steps.csv', index=False)
    pd.concat(matched, ignore_index=True).to_csv(OUT / 'e4_matched.csv', index=False)
    pd.concat(control, ignore_index=True).to_csv(OUT / 'e5_control.csv', index=False)
    pd.concat(holdout, ignore_index=True).to_csv(OUT / 'e10_holdout.csv', index=False)
    pd.concat(masks, ignore_index=True).to_csv(OUT / 'e10_mask.csv', index=False)
    cols = ['exp', 'feeder', 'Backbone', 'Scenario', 'Seed', 'method', 'ROC_forget', 'ROC_retain', 'Macro_ROC',
            'MIA_forget', 'gap_ROC_forget', 'gap_Macro_ROC', 'gap_MIA_forget', 'fidelity', 'probdiff',
            'Time', 'Epochs_run', 'Holdout_type', 'Forget_semantics', 'Seq_start']
    allr = pd.concat(keep, ignore_index=True)
    allr[[c for c in cols if c in allr]].to_csv(OUT / 'e4e10_trials.csv', index=False)
    (OUT / 'analysis_e4e10.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
