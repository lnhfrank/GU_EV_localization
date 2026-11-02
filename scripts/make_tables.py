"""Tables of the revised manuscript and the response letter, from the raw results.

Usage:
    python scripts/make_tables.py --check-submitted   # rebuild Table II of the submission, compare with the tex
    python scripts/make_tables.py [--out results/2026-10-20_tables] [--tex Paper_2026_ITSG/tables]

Sources (under results/, which git ignores):
  E3   2026-10-01_E3/{main,quiet,rest}_<bus>: the main-table methods except FT 10, 50 and 100 epochs, 34-bus and
       123-bus, seeds 42 to 51. The Retrain and Retrain_b rows of 34-bus GCN S3 s47 and GIN S3 s49 come from
       redo_cap_34bus, rerun after the old 600-epoch cap.
  E3ft 2026-10-12_E3ft/main_<bus>: FT 10, 50 and 100 epochs on every E3 trial.
  8500 2026-10-15_8500/{quiet,main}_8500bus: every method, S1 and S7, GCN and GIN, seeds 42 to 46, and
       2026-10-29_8500_s47-51/main_8500bus: the same for seeds 47 to 51, with the same code (e5d3c92).
Per trial (feeder, scenario, backbone, seed) and method: the test metrics of the result rows, the signed gap to
full-retraining (FR, Retrain) of the same trial, and the distances on the forgotten stations to Retrain_b, the
same retraining with another seed (seed + 1000): the absolute differences in ROC-AUC and MIA-AUC, the Fidelity
(share of test graphs and forgotten stations whose decision at 0.5 agrees) and the mean absolute probability
difference, from the saved per-graph logits. FR shares the seed, so the initialization and the batch order, with
the original model, and a model derived from the original model can be closer to FR than any independent
retraining is: on the 8500-node feeder the power-removed original model agrees with FR on 71% of the decisions
and with Retrain_b on 51% (S1). Distances are therefore measured to Retrain_b, and the distance of FR itself to
Retrain_b is the reference band. Original is scored with the charging power, every other method with
the withdrawn power removed, as in the runs.
Timing: quiet phase only (one process per GPU), S1, seeds 42 to 44, as in analysis_ft.py (E3ft) and
analysis_8500.py; for 123-bus GIN s42 and GCN s44 the E3 times come from quiet_redo_123bus.
Writes trials.csv, timing.csv, one csv per table and tables.txt to --out, and the LaTeX tables to --tex.
"""
import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import fit_scaler, stratified_split_multilabel  # noqa: E402
RES = ROOT / 'results'
E3, E3FT, E8500 = RES / '2026-10-01_E3', RES / '2026-10-12_E3ft', RES / '2026-10-15_8500'
E8500B = RES / '2026-10-29_8500_s47-51'                     # 8500-node seeds 47 to 51
RUNS_8500 = [E8500 / 'quiet_8500bus', E8500 / 'main_8500bus', E8500B / 'main_8500bus']
ARCHIVE = RES / 'archived (no useful)' / '2026-06-02'
FEEDERS = {'34bus': 'IEEE 34-bus', '123bus': 'IEEE 123-bus', '8500bus': 'IEEE 8500-node'}
FULL = {'34bus': 'S3', '123bus': 'S5', '8500bus': 'S7'}        # every station withdrawn
KEY = ['feeder', 'Backbone', 'Scenario', 'Seed']
METRICS = ['Macro_ROC', 'Macro_F1', 'ROC_forget', 'ROC_retain', 'MIA_forget', 'F1_forget']   # F1_forget: Table III
DIST = ['ROC_forget', 'MIA_forget']     # metrics whose distance to Retrain_b the tables report
NAME = {'Finetune_pure': 'FT_e25', 'FTpure_e10': 'FT_e10', 'FTpure_e50': 'FT_e50', 'FTpure_e100': 'FT_e100',
        'GIF_FT': 'GIF+FT'}
# Rows of the main table, in order, with their labels. IDEA runs only at the noise its certification bound needs,
# named IDEA; the sigma 0.001 arm is left out, since without noise IDEA is GIF
METHODS = [('Original', 'Original (with power)'), ('Mask_only', 'Original (power removed)'), ('GIF', 'GIF'),
           ('IDEA_cert', 'IDEA'),
           ('GIF+FT', 'GIF + FT 25'), ('FT_e10', 'FT 10'), ('FT_e25', 'FT 25'), ('FT_e50', 'FT 50'),
           ('FT_e100', 'FT 100'), ('WarmStart', 'FT to convergence'), ('Retrain', 'FR'),
           ('Retrain_b', 'FR, another seed')]
LABEL = dict(METHODS)
COST_BACKBONE = 'GCN'      # the backbone of the cost table, run on all three feeders
BACKBONES = ['GAT', 'GCN', 'GIN']
# Rows the cost tables leave out: Original (with power) and the second FR, and Original (power removed), which runs
# no update and only copies the weights
NO_COST = ('Original', 'Mask_only', 'Retrain_b')
LETTER_ONLY = ['GIF+FT']    # rows the main-text tables leave out: GIF + FT 25 matches FT 25
E3_METHODS =['Original', 'Mask_only', 'GIF', 'IDEA_cert', 'GIF_FT', 'Finetune_pure', 'WarmStart',
              'Retrain', 'Retrain_b']
# Arms of the component ablation in E3 (config arms 1 to 7 and Finetune_pure), for the letter
ABLATION = [('Mask_only', 'Original (power removed)'), ('Correction_only', 'Correction only'),
            ('BN_only', 'BN recalibration only'), ('GDGU_submitted', 'GDGU as submitted'),
            ('GDGU_ls', 'Correction, line search + BN + FT 25'), ('Random_step', 'Random step + BN + FT 25'),
            ('Finetune_only', 'BN + FT 25'), ('Finetune_pure', 'FT 25'), ('Retrain_b', 'FR, another seed')]
E4E10 = RES / '2026-10-15_E4E10'
OLD8500 = RES / 'archived (no useful)' / '2026-05-09_21'   # the 8500-node runs behind the submitted Table II, cap 300
E9N = RES / '2026-10-06_E9'                                 # E9 part 4, the floor under voltage noise
NOISE_LEVELS = [('clean', 'outputs', 'no noise'), ('snr20', 'outputs_snr20', '20 dB'),
                ('snr10', 'outputs_snr10', '10 dB'), ('snr0', 'outputs_snr0', '0 dB')]
# Sensitivity of the correction step (E5, 34-bus, S1, seeds 42 to 46, each arm + BN + FT 25): a fixed step rho
# along -dg/|dg|, or the submitted rule dg/lambda capped at |step| = 1 (corrected sign)
SENS = RES / '2026-10-02_E5' / 'sens_34bus'
SENS_ARMS = [(f'GDGU_rho{r}', 'Fixed step', r'$\rho$', r) for r in ('0.01', '0.03', '0.1', '0.3', '1')] + \
            [(f'GDGU_clip_lam{v}', 'Clip rule', r'$\lambda$', v) for v in ('0.01', '0.1', '1', '10', '100')]
COLUMNS = [(bus, scen) for bus in FEEDERS for scen in ('S1', FULL[bus])]


# ── Submitted Table II ──────────────────────────────────────────────────────────────────────────────────────────

def submitted_tex_rows():
    """(feeder, backbone, method) -> the five cells of Table II in GDGU_EV_loc.tex, as printed."""
    tex = (ROOT / 'Paper_2026_ITSG' / 'GDGU_EV_loc.tex').read_text()
    body = tex[tex.index(r'\label{tab:main}'):]
    body = body[:body.index(r'\end{tabular}')]
    feeder = backbone = None
    out = {}
    for line in body.splitlines():
        m = re.search(r'\\multirow\{\d+\}\{\*\}\{IEEE ([0-9]+)-(?:bus|node)\}', line)
        if m:
            feeder = f'{m.group(1)}bus'
        m = re.search(r'\\multirow\{\d+\}\{\*\}\{(GAT|GCN|GIN)\}', line)
        if m:
            backbone = m.group(1)
        cells = [c.strip() for c in line.rstrip('\\ ').split('&')]
        if len(cells) == 8 and cells[2] in ('Original', 'Retrain', 'GDGU', 'GIF', 'IDEA'):
            clean = [re.sub(r'\\textbf\{(.*)\}|\\underline\{(.*)\}', lambda x: x.group(1) or x.group(2), c)
                     for c in cells[3:]]
            out[(feeder, backbone, cells[2])] = clean
    return out


def check_submitted():
    """Rebuild Table II from the archived submission results with its conventions and compare every cell."""
    tex = submitted_tex_rows()
    rebuilt, bad = {}, []
    for bus in FEEDERS:
        d = pd.read_csv(ARCHIVE / f'{bus}_raw.csv')
        for (bb, m), g in d.groupby(['Backbone', 'Method']):
            if m not in ('Original', 'Retrain', 'GDGU', 'GIF', 'IDEA'):
                continue
            t_fr = d[(d.Backbone == bb) & (d.Method == 'Retrain')].Time.mean()

            def pm(c):     # mean and population sd over every scenario and seed, as in the submission
                return f'{g[c].mean():.3f}$\\pm${g[c].std(ddof=0):.3f}'
            rebuilt[(bus, bb, m)] = [pm('Macro_ROC'), pm('Macro_F1'),
                                     '--' if m == 'Original' else pm('MIA_forget'),
                                     '--' if m == 'Original' else f'{t_fr / g.Time.mean():.1f}$\\times$',
                                     f'{g.Peak_Memory_MB.mean():.0f}']
    for k, cells in tex.items():
        got = rebuilt.get(k)
        for i, (a, b) in enumerate(zip(cells, got or [None] * 5)):
            if a != b:
                bad.append((k, ['ROC-AUC', 'F1', 'MIA-AUC', 'Speedup', 'Mem'][i], a, b))
    n = 5 * len(tex)
    print(f'Table II of the submission: {len(tex)} rows, {n} cells; rebuilt from {ARCHIVE.name}: '
          f'{n - len(bad)} match, {len(bad)} differ')
    for k, col, a, b in bad:
        print(f'   {k} {col}: tex {a}  rebuilt {b}')
    return bad


# ── Trials of the revision ──────────────────────────────────────────────────────────────────────────────────────

def raw(folders):
    files = [f for d in folders for f in sorted(glob.glob(str(d / '*_raw.csv')))]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def outputs(folders):
    """(backbone, scenario, seed) -> path of the saved per-graph outputs, the later folder winning."""
    found = {}
    for d in folders:
        for f in sorted(glob.glob(str(d / 'outputs' / '*.pt'))):
            bb, scen, seed = Path(f).stem.split('_')
            found[(bb, scen, int(seed))] = Path(f)
    return found


def agreement(bus, files, redo=None, fr_files=None):
    """Forgotten-station Fidelity and probability difference to Retrain_b, and to FR, for every method of every trial.

    redo: outputs whose Retrain and Retrain_b logits replace those of files (FR rerun after the old cap).
    fr_files: outputs that hold FR for runs without it (E3ft ran the FT arms only); the test sets must match.
    """
    out = []
    for (bb, scen, seed), f in sorted(files.items()):
        saved = torch.load(f, map_location='cpu', weights_only=False)
        logits = saved['logits']
        for src in (fr_files, redo):
            if src and (bb, scen, seed) in src:
                ref = torch.load(src[(bb, scen, seed)], map_location='cpu', weights_only=False)
                assert np.array_equal(np.asarray(ref['idx_test']), np.asarray(saved['idx_test'])), (bb, scen, seed)
                logits.update({m: v for m, v in ref['logits'].items() if m in ('Retrain', 'Retrain_b')})
        forget = list(range(int(scen[1:])))

        def prob(m, part):
            return torch.sigmoid(torch.as_tensor(logits[m][part])).numpy()[:, forget]
        p_fr, p_b = prob('Retrain', 'test_occ'), prob('Retrain_b', 'test_occ')
        probs = {NAME.get(m, m): prob(m, 'test_occ') for m in logits if 'test_occ' in logits[m]}
        if 'Original' in logits:
            probs['Mask_only'] = probs.pop('Original')      # the original model without the withdrawn power
            probs['Original'] = prob('Original', 'test')    # and with it
        for m, p in probs.items():
            out.append(dict(feeder=bus, Backbone=bb, Scenario=scen, Seed=seed, method=m,
                            fidelity=float(((p > 0.5) == (p_b > 0.5)).mean()),
                            probdiff=float(np.abs(p - p_b).mean()),
                            fidelity_FR=float(((p > 0.5) == (p_fr > 0.5)).mean()),
                            probdiff_FR=float(np.abs(p - p_fr).mean()),
                            alarm=float((p > 0.5).mean())))   # share of the decisions that raise an alarm
    return pd.DataFrame(out)


def trials():
    """Every trial and method of the main table with metrics, gaps to FR and agreement with FR."""
    parts = []
    for bus in ('34bus', '123bus'):
        e3 = raw([E3 / f'{d}_{bus}' for d in ('main', 'quiet', 'rest')])
        e3 = e3[e3.Method.isin(E3_METHODS)]
        redo_files = None
        if bus == '34bus':
            redo = raw([E3 / 'redo_cap_34bus'])
            redo = redo[redo.Method.isin(['Retrain', 'Retrain_b'])]
            k = ['Backbone', 'Scenario', 'Seed', 'Method']
            e3 = pd.concat([e3.merge(redo[k], on=k, how='left', indicator=True).query('_merge == "left_only"')
                            .drop(columns='_merge'), redo], ignore_index=True)
            redo_files = outputs([E3 / 'redo_cap_34bus'])
        ft = raw([E3FT / f'main_{bus}'])
        ft = ft[ft.Method.isin(['FTpure_e10', 'FTpure_e50', 'FTpure_e100'])]
        df = pd.concat([e3, ft], ignore_index=True).assign(feeder=bus)
        files = outputs([E3 / f'{d}_{bus}' for d in ('main', 'quiet', 'rest')])
        files_ft = outputs([E3FT / f'main_{bus}'])
        agr = agreement(bus, files, redo_files)
        agr_ft = agreement(bus, files_ft, redo_files, fr_files=files)
        agr = pd.concat([agr, agr_ft[agr_ft.method.isin(['FT_e10', 'FT_e50', 'FT_e100'])]], ignore_index=True)
        parts.append((df, agr))
    df8 = raw(RUNS_8500).assign(feeder='8500bus')
    agr8 = agreement('8500bus', outputs(RUNS_8500))
    parts.append((df8, agr8))
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    agr = pd.concat([p[1] for p in parts], ignore_index=True)
    df['method'] = df.Method.map(lambda m: NAME.get(m, m))
    df = df[df.method.isin(LABEL)]
    dup = df.duplicated(KEY + ['method'])
    assert not dup.any(), df[dup][KEY + ['method']].head()
    fr = df[df.method == 'Retrain'].set_index(KEY)[METRICS]
    df = df.join(fr, on=KEY, rsuffix='_FR')
    for c in METRICS:
        df[f'gap_{c}'] = df[c] - df[f'{c}_FR']
    frb = df[df.method == 'Retrain_b'].set_index(KEY)[DIST]
    df = df.join(frb, on=KEY, rsuffix='_FRb')
    for c in DIST:
        df[f'dist_{c}'] = (df[c] - df[f'{c}_FRb']).abs()
    df = df.merge(agr, on=KEY + ['method'], how='left')
    keep = (KEY + ['method'] + METRICS + [f'gap_{c}' for c in METRICS] + [f'dist_{c}' for c in DIST]
            + ['fidelity', 'probdiff', 'fidelity_FR', 'probdiff_FR', 'alarm', 'Time', 'Epochs_run', 'Peak_Memory_MB'])
    return df[keep].sort_values(KEY + ['method']).reset_index(drop=True)


def timing():
    """Quiet-phase seconds per request: S1, seeds 42 to 44, one process per GPU."""
    rows = []
    for bus in ('34bus', '123bus'):
        q3 = raw([E3 / f'quiet_{bus}'])
        q3 = q3[q3.Method.isin(E3_METHODS) & (q3.Method != 'Finetune_pure')]   # FT comes from E3ft's quiet runs
        if bus == '123bus':
            redo = raw([E3 / 'quiet_redo_123bus'])
            k = ['Backbone', 'Seed', 'Method']
            q3 = pd.concat([q3.merge(redo[k], on=k, how='left', indicator=True).query('_merge == "left_only"')
                            .drop(columns='_merge'),
                            redo[redo.Method.isin(E3_METHODS) & (redo.Method != 'Finetune_pure')]], ignore_index=True)
        qft = raw([E3FT / f'quiet_{bus}'])
        qft = qft[qft.Method.isin(['Finetune_pure', 'FTpure_e10', 'FTpure_e50', 'FTpure_e100'])]
        rows.append(pd.concat([q3, qft], ignore_index=True).assign(feeder=bus))
    rows.append(raw([E8500 / 'quiet_8500bus']).assign(feeder='8500bus'))
    q = pd.concat(rows, ignore_index=True)
    q = q[(q.Scenario == 'S1') & q.Seed.between(42, 44)]
    q['method'] = q.Method.map(lambda m: NAME.get(m, m))
    q = q[q.method.isin(LABEL)]
    dup = q.duplicated(KEY + ['method'])
    assert not dup.any(), q[dup][KEY + ['method']].head()
    return q[KEY + ['method', 'Time', 'Epochs_run', 'Peak_Memory_MB']].reset_index(drop=True)


def signed(v, digits=3):
    """Signed number for a LaTeX table: a math minus, and a rounded zero shown as +0.000 rather than -0.000."""
    out = f'{v:+.{digits}f}'
    if float(out) == 0:
        out = out.replace('-', '+')
    return out.replace('-', '$-$')


def e3_raw(bus):
    """Every arm of E3 on a feeder, with the two Retrain rows of 34-bus rerun after the old cap replaced."""
    e3 = raw([E3 / f'{d}_{bus}' for d in ('main', 'quiet', 'rest')])
    if bus == '34bus':
        redo = raw([E3 / 'redo_cap_34bus'])
        redo = redo[redo.Method.isin(['Retrain', 'Retrain_b'])]
        k = ['Backbone', 'Scenario', 'Seed', 'Method']
        e3 = pd.concat([e3.merge(redo[k], on=k, how='left', indicator=True).query('_merge == "left_only"')
                        .drop(columns='_merge'), redo], ignore_index=True)
    return e3.assign(feeder=bus)


def ablation_trials():
    """The arms of the component ablation with their gaps to FR of the same trial."""
    out = []
    for bus in ('34bus', '123bus'):
        df = e3_raw(bus)
        df = df[df.Method.isin([m for m, _ in ABLATION] + ['Retrain'])].rename(columns={'Method': 'method'})
        assert not df.duplicated(KEY + ['method']).any()
        fr = df[df.method == 'Retrain'].set_index(KEY)[METRICS]
        df = df.join(fr, on=KEY, rsuffix='_FR')
        for c in METRICS:
            df[f'gap_{c}'] = df[c] - df[f'{c}_FR']
        out.append(df[KEY + ['method'] + METRICS + [f'gap_{c}' for c in METRICS]])
    return pd.concat(out, ignore_index=True)


def sensitivity():
    """The step arms of E5 against BN + FT 25 and FR of E3 on the same trials (34-bus, S1, seeds 42 to 46)."""
    s = raw([SENS])
    s = s[s.Method.isin([a for a, *_ in SENS_ARMS])]
    e3 = e3_raw('34bus')
    e3 = e3[(e3.Scenario == 'S1') & e3.Seed.between(42, 46) & e3.Method.isin(['Finetune_only', 'Finetune_pure', 'Retrain'])]
    df = pd.concat([s, e3], ignore_index=True).rename(columns={'Method': 'method'}).assign(feeder='34bus')
    assert not df.duplicated(KEY + ['method']).any()
    fr = df[df.method == 'Retrain'].set_index(KEY)[METRICS]
    bn = df[df.method == 'Finetune_only'].set_index(KEY)
    df = df.join(fr, on=KEY, rsuffix='_FR')
    rows, per_trial = [], []
    for arm, kind, sym, val in SENS_ARMS + [('Finetune_only', 'BN + FT 25, no correction', '', ''),
                                            ('Finetune_pure', 'FT 25, no BN', '', '')]:
        g = df[df.method == arm].set_index(KEY)
        d = g.Macro_ROC - bn.reindex(g.index).Macro_ROC
        per_trial.append(pd.DataFrame(dict(
            arm=arm, kind=kind, value=val, step=g.Step_len if 'Step_len' in g else np.nan,
            vs_bn_Macro_ROC=d, vs_bn_ROC_forget=g.ROC_forget - bn.reindex(g.index).ROC_forget,
            vs_bn_MIA_forget=g.MIA_forget - bn.reindex(g.index).MIA_forget)).reset_index())
        rows.append(dict(arm=arm, kind=kind, symbol=sym, value=val, n=len(g),
                         step=g.Step_len.median() if 'Step_len' in g else np.nan,
                         clipped=(g.Dg_norm / float(val) > 1.0).mean() if kind == 'Clip rule' else np.nan,
                         macro_gap=(g.Macro_ROC - g.Macro_ROC_FR).mean(),
                         forget_gap=(g.ROC_forget - g.ROC_forget_FR).mean(),
                         mia_gap=(g.MIA_forget - g.MIA_forget_FR).mean(),
                         vs_bn=d.mean(), p_vs_bn=wilcoxon_p(d)))
    return pd.DataFrame(rows), pd.concat(per_trial, ignore_index=True)


def wilcoxon_p(d, alternative='two-sided'):
    """p of the Wilcoxon signed-rank test of paired differences d; NaN when there is nothing to test."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.all(d == 0):
        return np.nan
    return float(wilcoxon(d, alternative=alternative).pvalue)


def scenarios(T):
    """Every (feeder, scenario) of the trials, in feeder order and by the number of stations withdrawn."""
    return [(bus, s) for bus in FEEDERS for s in sorted(T[T.feeder == bus].Scenario.unique(), key=lambda x: int(x[1:]))]


def paired_stats(T, methods):
    """Per feeder, scenario (every scenario, R1-B3) and method, paired over the trials (backbone x seed).

    Signed gaps are to FR; distances and Fidelity are to Retrain_b, against those of FR itself (the band).
    """
    rows = []
    for bus, scen in scenarios(T):
        s = T[(T.feeder == bus) & (T.Scenario == scen)]
        band = s[s.method == 'Retrain'].set_index(KEY)
        for m in methods:
            g = s[s.method == m].set_index(KEY)
            if g.empty:
                continue
            b = band.reindex(g.index)
            rows.append(dict(
                feeder=bus, Scenario=scen, method=m, n=len(g),
                macro_gap=g.gap_Macro_ROC.mean(), p_macro=wilcoxon_p(g.gap_Macro_ROC),
                retained_gap=g.gap_ROC_retain.mean(),
                forget_gap=g.gap_ROC_forget.mean(), p_forget=wilcoxon_p(g.gap_ROC_forget),
                dist_forget=g.dist_ROC_forget.mean(), band_dist=b.dist_ROC_forget.mean(),
                p_band=wilcoxon_p(g.dist_ROC_forget - b.dist_ROC_forget, 'greater'),
                fidelity=g.fidelity.mean(), band_fidelity=b.fidelity.mean(),
                p_band_fidelity=wilcoxon_p(b.fidelity - g.fidelity, 'greater')))
    return pd.DataFrame(rows)


def control_cells(A, arm, ctrl, col):
    """Per backbone x scenario cell, the mean difference arm - ctrl in col and its two-sided Wilcoxon p, paired by seed."""
    k = ['Backbone', 'Scenario', 'Seed']
    a = A[A.method == arm].set_index(k)[col]
    b = A[A.method == ctrl].set_index(k)[col].reindex(a.index)
    d = (a - b).rename('d').reset_index()
    return d.groupby(['Backbone', 'Scenario']).d.agg(['mean', wilcoxon_p]).rename(columns={'wilcoxon_p': 'p'})


def preregistered(A):
    """The pre-registered rule per backbone x scenario cell: GDGU_ls against BN + FT 25, paired by seed."""
    rows = []
    k = ['Backbone', 'Scenario', 'Seed']
    for (bb, scen), g in A.groupby(['Backbone', 'Scenario']):
        a = g[g.method == 'GDGU_ls'].set_index(k)
        b = g[g.method == 'Finetune_only'].set_index(k).reindex(a.index)
        d = a.Macro_ROC - b.Macro_ROC
        rows.append(dict(Backbone=bb, Scenario=scen, gain=d.mean(), p=wilcoxon_p(d, 'greater'),
                         gain_forget=(a.ROC_forget - b.ROC_forget).mean(), gain_mia=(a.MIA_forget - b.MIA_forget).mean()))
    c = pd.DataFrame(rows)
    c['counts'] = (c.gain > 0.005) & (c.p < 0.05)
    return c


def e4_e10_tables(args, log):
    """Sequential requests (E4) and unseen attack types (E10a), for the letter."""
    st = pd.read_csv(E4E10 / 'e4_steps.csv')
    rows = []
    for bus in ('34bus', '123bus'):
        s = st[st.feeder == bus]
        for j in sorted(s.step.unique()):
            r = {'feeder': bus, 'step': j}
            for m in ('FT_e25', 'FT_e100'):
                for mode, tag in (('sequential', 'seq'), ('one-shot', 'non-seq')):
                    g = s[(s.step == j) & (s['mode'] == mode) & (s.method == m)]
                    r[f'{m} {tag} ROC'] = g.gap_ROC_forget.mean()
                    r[f'{m} {tag} MIA'] = g.gap_MIA_forget.mean()
            b = s[(s.step == j) & (s['mode'] == 'sequential') & (s.method == 'Retrain_b')]
            r['band ROC'], r['band MIA'] = b.gap_ROC_forget.abs().mean(), b.gap_MIA_forget.abs().mean()
            rows.append(r)
    e4 = pd.DataFrame(rows)
    e4.to_csv(args.out / 'table_e4.csv', index=False)
    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{Sequential and non-sequential requests (E4): gap to FR of the same step in the '
           r'forgotten-station ROC-AUC / MIA-AUC, mean over three backbones and seeds 42 to 46. Band: mean '
           r'$|$Retrain$_b$ $-$ Retrain$|$.}', r'\label{tab:e4}', r'\footnotesize', r'\setlength{\tabcolsep}{2pt}',
           r'\begin{tabular}{llccccc}', r'\toprule',
           mr('Feeder') + ' & ' + mr('Step') + r' & \multicolumn{2}{c}{FT 25} & \multicolumn{2}{c}{FT 100} & '
           + mr('Band') + r' \\',
           r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}', r' & & Seq. & Non-seq. & Seq. & Non-seq. & \\', r'\midrule']
    for _, r in e4.iterrows():
        cells = [f'{signed(r[f"{m} {t} ROC"])} / {signed(r[f"{m} {t} MIA"])}' for m in ('FT_e25', 'FT_e100')
                 for t in ('seq', 'non-seq')]
        tex.append(f'{FEEDERS[r.feeder].replace("IEEE ", "")} & S{r.step} & ' + ' & '.join(cells)
                   + f' & {r["band ROC"]:.3f} / {r["band MIA"]:.3f}' + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (args.tex / 'table_e4.tex').write_text('\n'.join(tex) + '\n')

    h = pd.read_csv(E4E10 / 'e10_holdout.csv', dtype={'type': str})
    h = h[(h.method == 'Retrain') & h['type'].isin(['1', '2', '3', '4'])]
    g = h.groupby(['feeder', 'type', 'Scenario', 'training']).ROC_forget.mean().unstack('training')
    g.to_csv(args.out / 'table_e10a.csv')
    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{Leakage floor for an attack type seen or not seen in training (E10a): forgotten-station ROC-AUC '
           r'of FR, not seen / seen, GCN, mean over seeds 42 to 46.}', r'\label{tab:e10a}', r'\footnotesize',
           r'\setlength{\tabcolsep}{2.5pt}', r'\begin{tabular}{llccccc}', r'\toprule',
           r'Feeder & Type & S1 & S2 & S3 & S4 & S5 \\', r'\midrule']
    for bus in ('34bus', '123bus'):
        for t in ('1', '2', '3', '4'):
            cells = []
            for j in range(1, 6):
                key = (bus, t, f'S{j}')
                cells.append(f'{g.loc[key, "held out"]:.3f} / {g.loc[key, "seen"]:.3f}'
                             if key in g.index and not np.isnan(g.loc[key, 'held out']) else '--')
            tex.append(f'{FEEDERS[bus].replace("IEEE ", "")} & {t} & ' + ' & '.join(cells) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (args.tex / 'table_e10a.tex').write_text('\n'.join(tex) + '\n')
    for bus in ('34bus', '123bus'):
        x = g.loc[bus]
        x = x[x['held out'].notna()]
        d = x['held out'] - x['seen']
        t1 = x.index.get_level_values('type') == '1'
        log(f'== E10a {bus}: not seen Types 2 to 4 {x["held out"][~t1].min():.3f} to {x["held out"][~t1].max():.3f} '
            f'(minus seen {d[~t1].min():+.3f} to {d[~t1].max():+.3f}); Type 1 {x["held out"][t1].min():.3f} to '
            f'{x["held out"][t1].max():.3f} (minus seen {d[t1].min():+.3f} to {d[t1].max():+.3f})')


def letter_numbers(log):
    """Numbers the response letter quotes outside the tables, each recomputed from the raw results."""
    cfg = {bus: yaml.safe_load((ROOT / 'config' / f'{bus}.yaml').read_text()) for bus in ('34bus', '123bus')}
    k = ['Backbone', 'Scenario', 'Seed']

    log('== letter, R1-B1 and R3-3: GDGU as submitted against the corrected GDGU with line search (E3), Macro ROC, '
        'paired by seed in each backbone x scenario cell, two-sided Wilcoxon')
    for bus in ('34bus', '123bus'):
        d = e3_raw(bus)
        a = d[d.Method == 'GDGU_submitted'].set_index(k)
        b = d[d.Method == 'GDGU_ls'].set_index(k).reindex(a.index)
        diff = (a.Macro_ROC - b.Macro_ROC).rename('d').reset_index()
        cells = diff.groupby(['Backbone', 'Scenario']).d.agg(['mean', wilcoxon_p])
        sig = cells.wilcoxon_p < 0.05
        log(f'   {bus}: mean difference {diff.d.mean():+.4f}; cell means {cells["mean"].min():+.4f} to '
            f'{cells["mean"].max():+.4f}; p < 0.05 in {sig.sum()}/{len(cells)} cells, all of them lower: '
            f'{bool((cells["mean"][sig] < 0).all())}')

    log('== letter, R1-RA10 and R3-3: the clip rule of the submission, step dg / lambda capped at norm rho '
        '(E3, GDGU as submitted, every trial)')
    for bus in ('34bus', '123bus'):
        lam, rho = cfg[bus]['gdgu']['damp'], cfg[bus]['gdgu']['max_norm']
        s = e3_raw(bus)
        s = s[s.Method == 'GDGU_submitted']
        ratio = s.Dg_norm / lam / rho
        clipped = ratio > 1
        assert np.allclose(1 / s.Clip_coef[clipped], ratio[clipped], rtol=1e-5)
        log(f'   {bus}: lambda {lam}, rho {rho}; the clip acts in {clipped.sum()}/{len(s)} requests; |dg| / (lambda rho) '
            f'{ratio.min():.0f} to {ratio.max():.0f}, median {ratio.median():.0f}')
        # per scenario, as R3-3 asks; |dg| / lambda equals |dg| / (lambda rho) at rho = 1
        for scen in sorted(s.Scenario.unique(), key=lambda x: int(x[1:])):
            r = (s.Dg_norm / lam)[s.Scenario == scen]
            log(f'      {scen}: |dg| / lambda {r.min():.0f} to {r.max():.0f}, median {r.median():.0f}; the clip acts in '
                f'{(r > rho).sum()}/{len(r)} requests')

    log('== letter, R1-3(v), R2-11 and R2-12: peak GPU memory of the influence update (E3 quiet phase, S1, seeds 42 '
        'to 44): the submitted HVP, one graph over its 11 or 16 batches (GIF_fullHVP), against one batch at a time (GIF)')
    for bus in ('34bus', '123bus'):
        q = raw([E3 / f'quiet_{bus}'])
        q = q[(q.Scenario == 'S1') & q.Seed.between(42, 44)]
        mem = q.pivot_table(index=['Backbone', 'Seed'], columns='Method', values='Peak_Memory_MB')
        r = mem.GIF_fullHVP / mem.GIF
        log(f'   {bus}: {len(mem)} trials; submitted {mem.GIF_fullHVP.min():.0f} to {mem.GIF_fullHVP.max():.0f} MB, per batch '
            f'{mem.GIF.min():.0f} to {mem.GIF.max():.0f} MB, {r.min():.1f} to {r.max():.1f} times lower; FT 25 '
            f'{mem.Finetune_pure.min():.0f} to {mem.Finetune_pure.max():.0f} MB; largest for '
            f'{mem.GIF_fullHVP.groupby(level="Backbone").max().idxmax()}')

    log('== letter, R2-5: noise sigma of IDEA (the bound of OpenGU, eps 1, delta 1e-4, its default constants) '
        'against the RMS of the weights of the original model')
    for bus, rows, caches in (('34bus', e3_raw('34bus'), [E3 / 'cache_34bus_e3000']),
                              ('123bus', e3_raw('123bus'), [E3 / 'cache_123bus_e3000']),
                              ('8500bus', raw(RUNS_8500), [E8500 / 'cache_8500', E8500B / 'cache_8500'])):
        c = rows[rows.Method == 'IDEA_cert']
        one, full = c[c.Scenario == 'S1'].Noise_std, c[c.Scenario == FULL[bus]].Noise_std
        rms = []
        for f in sorted(f for d in caches for f in glob.glob(str(d / 'Original_*.pt'))):
            sd = torch.load(f, map_location='cpu', weights_only=False)['state_dict']
            w = torch.cat([v.float().flatten() for n, v in sd.items() if v.is_floating_point() and 'running' not in n])
            rms.append(float(w.pow(2).mean().sqrt()))
        log(f'   {bus}: sigma {c.Noise_std.min():.2f} to {c.Noise_std.max():.2f} (S1 {one.min():.2f} to {one.max():.2f}, '
            f'every station {full.min():.2f} to {full.max():.2f}); weight RMS {min(rms):.3f} to {max(rms):.3f} over '
            f'{len(rms)} original models')

    log('== letter, R2-4 and R3-6: zeroing the raw power before standardization with the scaler fit on the training '
        'graphs (seeds 42 to 51): per station, the steps whose power is constant over the training graphs (the scaler '
        'keeps scale 1), the steps whose spread is nonzero but below 1e-3 kW, and the standardized zero')
    for bus in ('34bus', '123bus'):
        path, = glob.glob(str(E3 / 'data_cache' / f'{bus}_raw_*.pkl'))
        with open(path, 'rb') as fh:
            d = pickle.load(fh)
        rows = []
        for seed in range(42, 52):
            idx_train, _, _ = stratified_split_multilabel(d['all_y'], cfg[bus]['training']['test_size'],
                                                          cfg[bus]['training']['val_ratio'], seed)
            for node in d['evcs_node_indices']:
                sc = fit_scaler(d['all_x_P'][:, [node], :], idx_train)   # the columns of this station, as in the runs
                std = np.sqrt(sc.var_)
                rows.append(dict(const=int((std == 0).sum()), low=int(((std > 0) & (std < 1e-3)).sum()),
                                 smallest=std[std > 0].min(), zero=float((-sc.mean_ / sc.scale_).min())))
        r = pd.DataFrame(rows)
        log(f'   {bus}: of {d["n_feat"]} steps per station, {r.const.min()} to {r.const.max()} constant and {r.low.min()} '
            f'to {r.low.max()} with a spread below 1e-3 kW, the smallest {r.smallest.min():.1e} kW; the standardized '
            f'zero reaches {r.zero.min():.1e}')

    if not (ARCHIVE.exists() and OLD8500.exists()):   # the archived runs of the submission, for the letter
        return
    log('== letter, R1-3(iii): the 8500-node runs of the submission, trials whose training ran to the cap')
    sub = pd.read_csv(ARCHIVE / '8500bus_raw.csv')
    for bb in ('GCN', 'GIN'):
        s = pd.read_csv(OLD8500 / f'8500bus_{bb}_summary.csv', header=[0, 1], index_col=[0, 1, 2])
        t = s[('Time', 'mean')].droplevel(0)
        t.index = t.index.set_names(['Scenario', 'Method'])
        same = sub[sub.Backbone == bb].groupby(['Scenario', 'Method']).Time.mean().reindex(t.index)
        assert np.allclose(t, same, atol=1e-3), f'{OLD8500.name} is not the run of the submitted table ({bb})'
        L = json.loads((OLD8500 / f'8500bus_{bb}_epoch_logs.json').read_text())
        cap = L['_metadata']['config']['epochs']
        n = {m: [len(v[m]) >= cap for key, v in L.items() if key != '_metadata'] for m in ('Original', 'Retrain')}
        log(f'   {bb} (cap {cap}, {OLD8500.name}, the run of the submitted table): ' + ', '.join(
            f'{m} {sum(v)}/{len(v)} ({sum(v) / len(v):.0%})' for m, v in n.items()))


def run_agreement(files):
    """Forgotten-station Fidelity and probability difference to Retrain_b and to FR in runs that save every method's
    test logits with the withdrawn power removed (E4 sequential, E9 part 4); Original is left out."""
    out = []
    for f in files:
        stem = Path(f).stem.split('_')
        bb, scen, seed = stem[0], stem[1], int(stem[-1])
        logits = torch.load(f, map_location='cpu', weights_only=False)['logits']
        forget = list(range(int(scen[1:])))
        p = {m: torch.sigmoid(torch.as_tensor(v['test_occ'])).numpy()[:, forget]
             for m, v in logits.items() if m != 'Original'}
        for m, pm in p.items():
            out.append(dict(Backbone=bb, Scenario=scen, Seed=seed, method=NAME.get(m, m),
                            fidelity=float(((pm > 0.5) == (p['Retrain_b'] > 0.5)).mean()),
                            probdiff=float(np.abs(pm - p['Retrain_b']).mean()),
                            fidelity_FR=float(((pm > 0.5) == (p['Retrain'] > 0.5)).mean())))
    return pd.DataFrame(out)


def agreement_e4_e9(log):
    """Fidelity in the sequential runs (E4) and under voltage noise (E9 part 4), to Retrain_b as in the tables."""
    def line(g, methods):
        band = g[g.method == 'Retrain']
        return (f'band {band.fidelity.mean():.3f} / {band.probdiff.mean():.3f}; ' + ', '.join(
            f'{m} {g[g.method == m].fidelity.mean():.3f} / {g[g.method == m].probdiff.mean():.3f} '
            f'(to FR {g[g.method == m].fidelity_FR.mean():.3f})' for m in methods if (g.method == m).any()))

    log('== E4 sequential requests: forgotten-station Fidelity / prob. diff to Retrain_b per step, FR\'s own the band '
        '(Fidelity to FR, the same seed, for comparison)')
    for bus in ('34bus', '123bus'):
        a = run_agreement(sorted(glob.glob(str(E4E10 / f'e4_{bus}' / 'outputs' / '*.pt'))))
        for scen in sorted(a.Scenario.unique(), key=lambda x: int(x[1:])):
            g = a[a.Scenario == scen]
            log(f'   {bus} {scen} (n {g[g.method == "Retrain"].shape[0]}): '
                + line(g, ['Mask_only', 'FT_e25', 'FT_e100', 'GIF+FT']))
    log('== E9 part 4, every station withdrawn under voltage noise: forgotten-station Fidelity / prob. diff to '
        'Retrain_b, FR\'s own the band (Fidelity to FR for comparison)')
    for bus in ('34bus', '123bus'):
        for lev, outdir, name in NOISE_LEVELS:
            g = run_agreement(sorted(glob.glob(str(E9N / f'{bus}_{lev}' / outdir / '*.pt'))))
            log(f'   {bus} {name} (n {g[g.method == "Retrain"].shape[0]}): '
                + line(g, ['Mask_only', 'GIF', 'FT_e25', 'FT_e50', 'FT_e100']))


# ── Tables ──────────────────────────────────────────────────────────────────────────────────────────────────────

def cell_table(T, value, methods):
    """methods x (feeder, scenario) of value(group of trials) -> string; '' where there are no trials."""
    out = pd.DataFrame(index=[m for m in methods], columns=pd.MultiIndex.from_tuples(COLUMNS), dtype=object)
    for m in methods:
        for bus, scen in COLUMNS:
            g = T[(T.method == m) & (T.feeder == bus) & (T.Scenario == scen)]
            out.loc[m, (bus, scen)] = value(g) if len(g) else ''
    return out


MAIN_COLOR = r'\color{blue}'    # tables new in the revision are blue throughout, as the revised tex marks them


def mr(text):
    """A header cell centered over the two header rows; -0.6ex makes up for the space of the cmidrule between them."""
    return rf'\multirow{{2}}{{*}}[-0.6ex]{{{text}}}'


def to_latex(tab, caption, label, note='', sep=5, color='', width='', rule_before=()):
    """A table* with methods in rows and two scenarios per feeder in columns; sep is the column separation in pt,
    color a color command for the whole table, as MAIN_COLOR for the main-text tables, and width a length such as
    0.94\\textwidth that the table fills, with the extra space spread between the columns and the note as wide as
    the table; rule_before lists the methods whose rows a midrule precedes, which sets the
    original models, the unlearning methods and FR apart."""
    heads = ' & '.join(rf'\multicolumn{{2}}{{c}}{{{FEEDERS[b]}}}' for b in FEEDERS)
    sub = ' & '.join(f'{s} (single)' if s == 'S1' else f'{s} (all)' for _, s in COLUMNS)
    rules = ' '.join(rf'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(FEEDERS)))
    cols = 'l' + 'c' * len(COLUMNS)
    begin, end = ((rf'\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}{cols}@{{}}}}', r'\end{tabular*}')
                  if width else (rf'\begin{{tabular}}{{{cols}}}', r'\end{tabular}'))
    lines = [r'\begin{table*}[t]', r'\centering'] + ([color] if color else []) + [
             rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\footnotesize',
             rf'\setlength{{\tabcolsep}}{{{sep}pt}}', begin, r'\toprule',
             rf'{mr("Method")} & {heads} \\', rules, rf' & {sub} \\', r'\midrule']
    for m, row in tab.iterrows():
        if m in rule_before:
            lines.append(r'\midrule')
        lines.append(f'{LABEL[m]} & ' + ' & '.join(v if v else '--' for v in row.values) + r' \\')
    lines += [r'\bottomrule', end]
    if note:
        box = width or r'\textwidth'
        lines.append(rf'\par\smallskip\parbox{{{box}}}{{\footnotesize {note}}}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--check-submitted', action='store_true')
    p.add_argument('--out', type=Path, default=RES / '2026-10-20_tables')
    p.add_argument('--tex', type=Path, default=ROOT / 'Paper_2026_ITSG' / 'tables')
    args = p.parse_args()
    if args.check_submitted:
        raise SystemExit(1 if check_submitted() else 0)

    args.out.mkdir(parents=True, exist_ok=True)
    args.tex.mkdir(parents=True, exist_ok=True)
    T = trials()
    Q = timing()
    T.to_csv(args.out / 'trials.csv', index=False)
    Q.to_csv(args.out / 'timing.csv', index=False)
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)
    n = T.groupby(['feeder', 'Scenario']).apply(lambda g: g[g.method == 'Retrain'].shape[0])
    log('Trials per feeder and scenario (backbones x seeds): ' + ', '.join(f'{b} {s} {v}' for (b, s), v in n.items()))
    methods = [m for m, _ in METHODS]
    band = T[T.method == 'Retrain_b']

    # 1. Forgotten-station ROC-AUC: the leakage floor is the FR row
    t1 = cell_table(T, lambda g: f'{g.ROC_forget.mean():.3f}$\\pm${g.ROC_forget.std():.3f}', methods)
    # 2. Fidelity / probability difference to Retrain_b on the forgotten stations; the FR row is the band
    near = [m for m in methods if m != 'Retrain_b']
    t2 = cell_table(T, lambda g: f'{g.fidelity.mean():.3f} / {g.probdiff.mean():.3f}', near)
    # 3. |ROC-AUC| / |MIA-AUC| differences to Retrain_b on the forgotten stations; the FR row is the band
    t3 = cell_table(T, lambda g: f'{g.dist_ROC_forget.mean():.3f} / '
                                 + ('--' if g.dist_MIA_forget.isna().all() else f'{g.dist_MIA_forget.mean():.3f}'), near)
    # 4. Probability difference to Retrain_b on the forgotten stations, for the letter
    t4 = cell_table(T, lambda g: f'{g.probdiff.mean():.3f}$\\pm${g.probdiff.std():.3f}', near)
    for name, tab in (('forgotten_roc', t1), ('agreement', t2), ('distance', t3), ('probdiff', t4)):
        tab.to_csv(args.out / f'table_{name}.csv')
    mean_over = ('Mean over the backbones (GAT, GCN and GIN; GCN and GIN on the 8500-node feeder) and seeds '
                 '42 to 51.')
    note = mean_over + ' S1 withdraws one station, the other column every station.'
    ref = ('FR shares the seed with the original model, so the reference is a second retraining '
           'with another seed, and the FR row, the distance between the two retrainings, is the '
           'reference band. ')
    # The main-text tables leave out LETTER_ONLY. The agreement table gives the fidelity alone as mean +- sd, as the
    # submitted table gave one metric per cell, and bolds one cell per column, the highest fidelity of the methods,
    # FR excluded; the paired tests against the band stay in tables.txt and the text, and the
    # probability difference goes to the letter (table_probdiff)
    shown = [m for m in near if m not in LETTER_ONLY]

    def fid(g):   # as printed, so that a tie in print is bold in both cells
        return float(f'{g.fidelity.mean():.3f}')
    best = {}
    for bus, scen in COLUMNS:
        s = T[(T.feeder == bus) & (T.Scenario == scen)]
        best[(bus, scen)] = max(fid(s[s.method == m]) for m in shown if m != 'Retrain' and (s.method == m).any())

    def agree(g):
        m, bus, scen = g.method.iloc[0], g.feeder.iloc[0], g.Scenario.iloc[0]
        cell = f'{fid(g):.3f}$\\pm${g.fidelity.std():.3f}'
        return rf'\textbf{{{cell}}}' if m != 'Retrain' and fid(g) == best[(bus, scen)] else cell
    t2_main = cell_table(T, agree, shown)
    # Table III reports the forgotten-station F1 score, a common metric that needs no second FR, and bolds per column
    # the GU method whose printed mean is closest to FR; the fidelity table goes to the letter
    def f1v(g):
        return float(f'{g.F1_forget.mean():.3f}')
    gu = [m for m in shown if m not in ('Original', 'Mask_only', 'Retrain')]
    near_fr = {}
    for bus, scen in COLUMNS:
        s = T[(T.feeder == bus) & (T.Scenario == scen)]
        fr_f1 = f1v(s[s.method == 'Retrain'])
        near_fr[(bus, scen)] = (fr_f1, min(abs(f1v(s[s.method == m]) - fr_f1) for m in gu if (s.method == m).any()))

    def f1cell(g):
        m, bus, scen = g.method.iloc[0], g.feeder.iloc[0], g.Scenario.iloc[0]
        cell = f'{f1v(g):.3f}$\\pm${g.F1_forget.std():.3f}'
        fr_f1, d = near_fr[(bus, scen)]
        return rf'\textbf{{{cell}}}' if m in gu and abs(abs(f1v(g) - fr_f1) - d) < 1e-9 else cell
    t_f1 = cell_table(T, f1cell, shown)
    bold = 'Bold marks the highest fidelity of the GU methods. '   # as in the tex
    (args.tex / 'table_forgotten_roc.tex').write_text(to_latex(
        t1.drop(index=LETTER_ONLY), 'Forgotten-station ROC-AUC of the original model, the unlearning methods and '
        'FR (mean$\\pm$sd). The FR row is the leakage floor.', 'tab:forgotten_roc', note,
        color=MAIN_COLOR))
    # Main-text captions: blue as their tables, no colons, EVCS for a single station; the
    # column heads (single) and (all) already say what the scenarios withdraw. The metric names of Metrics, fidelity
    # in lower case, the second FR as the reference, and which way each number improves.
    # Captions of the main-text tables start with The; the caption and the note of the agreement table as
    # shortened in the tex
    # Table III: the caption and the note as in the tex, so that a rerun does not undo them; the
    # threshold of 0.5 is stated in Metrics
    (args.tex / 'table_agreement.tex').write_text(to_latex(
        t_f1, r'\textcolor{blue}{The forgotten-station F1 score.}', 'tab:agreement',
        r'\textbf{Bold} marks the GU method closest to FR. All the numbers are mean$\pm$sd values over the GNN '
        r'backbones and 10 seeds.', color=MAIN_COLOR, width=r'0.94\textwidth',
        rule_before=('GIF', 'Retrain')))
    (args.tex / 'table_fidelity.tex').write_text(to_latex(   # the former Table III, for the letter
        t2_main, r'\textcolor{blue}{The fidelity on the forgotten EVCSs (mean$\pm$sd).}',
        'tab:fidelity',
        'A higher fidelity means a model closer to retraining. '
        'The reference is the second FR, and the FR row is the reference band. ' + bold
        + 'Mean over the GNN backbones and 10 seeds.', color=MAIN_COLOR, width=r'0.94\textwidth',
        rule_before=('GIF', 'Retrain')))
    (args.tex / 'table_distance.tex').write_text(to_latex(
        t3, 'Distance to full-retraining on the forgotten stations: mean absolute difference in ROC-AUC / '
        'MIA-AUC.', 'tab:distance', ref + note, sep=3))   # for the letter page
    (args.tex / 'table_probdiff.tex').write_text(to_latex(
        t4, 'Mean absolute probability difference to full-retraining on the forgotten stations (mean$\\pm$sd).',
        'tab:probdiff', ref + note, sep=3))   # for the letter page

    # 5. Cost per request: quiet-phase seconds (median) and peak memory (max) over seeds 42 to 44, S1. The cost depends
    # on the backbone and the 8500-node feeder has no GAT, so the paper reports GCN, the backbone of all three feeders,
    # and the letter every backbone
    def cost_table(q):
        rows = []
        for m in methods:
            row = {'method': m}
            for bus in FEEDERS:
                g = q[(q.method == m) & (q.feeder == bus)]
                row[f'{bus} s'] = g.Time.median() if len(g) else np.nan
                row[f'{bus} MB'] = g.Peak_Memory_MB.max() if len(g) else np.nan
            rows.append(row)
        return pd.DataFrame(rows).set_index('method')
    cost = cost_table(Q[Q.Backbone == COST_BACKBONE])
    cost.to_csv(args.out / 'table_cost.csv')
    cost_bb = {bb: cost_table(Q[Q.Backbone == bb]) for bb in BACKBONES}
    pd.concat(cost_bb, names=['Backbone']).to_csv(args.out / 'table_cost_backbones.csv')
    tex = [r'\begin{table}[t]', r'\centering', MAIN_COLOR,
           # the caption as in the tex; the seeds, the median and one process per GPU are in the text
           r'\caption{\textcolor{blue}{The time consumption and peak GPU memory usage per request in $\mathrm{S}_1$ '
           r'for GCN.}}',
           r'\label{tab:cost}', r'\footnotesize',
           r'\setlength{\tabcolsep}{3pt}',   # 94% of the column width, the extra space between the columns
           r'\begin{tabular*}{0.94\columnwidth}{@{\extracolsep{\fill}}l' + 'rr' * len(FEEDERS) + '@{}}', r'\toprule',
           mr('Method') + ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{FEEDERS[b].replace("IEEE ", "")}}}' for b in FEEDERS)
           + r' \\', ' '.join(rf'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(FEEDERS))),
           ' & ' + ' & '.join(['s & MB'] * len(FEEDERS)) + r' \\', r'\midrule']
    for m, r in cost.iterrows():
        if m in NO_COST or m in LETTER_ONLY:
            continue
        vals = []
        for bus in FEEDERS:
            s, mb = r[f'{bus} s'], r[f'{bus} MB']
            vals += ['--' if np.isnan(s) else (f'{s:.1f}' if s >= 0.05 else '$<$0.1'),
                     '--' if np.isnan(mb) else f'{mb:.0f}']
        tex.append(f'{LABEL[m]} & ' + ' & '.join(vals) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    (args.tex / 'table_cost.tex').write_text('\n'.join(tex) + '\n')
    # The same cost for every backbone, for the letter: time and memory blocks, one column per feeder and backbone
    cols = [(bus, bb) for bus in FEEDERS for bb in BACKBONES if (Q[(Q.feeder == bus) & (Q.Backbone == bb)]).shape[0]]
    tex = [r'\begin{table}[h]', r'\centering',
           r'\caption{The time consumption and peak GPU memory usage per request in $\mathrm{S}_1$ for every backbone.}',
           r'\label{tab:cost_backbones}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}l' + 'r' * len(cols) + '@{}}', r'\toprule',
           ' & ' + ' & '.join(rf'\multicolumn{{{sum(c[0] == b for c in cols)}}}{{c}}{{{FEEDERS[b].replace("IEEE ", "")}}}'
                              for b in FEEDERS) + r' \\']
    tex.append(' '.join(rf'\cmidrule(lr){{{2 + i}-{1 + i + sum(c[0] == b for c in cols)}}}'
                        for i, b in [(sum(c[0] in list(FEEDERS)[:k] for c in cols), b)
                                     for k, b in enumerate(FEEDERS)]))
    tex.append('Method & ' + ' & '.join(bb for _, bb in cols) + r' \\')
    for unit, fmt in (('s', lambda v: f'{v:.1f}' if v >= 0.05 else '$<$0.1'), ('MB', lambda v: f'{v:.0f}')):
        tex += [r'\midrule', rf'\multicolumn{{{1 + len(cols)}}}{{@{{}}l}}{{\emph{{{"Time (s)" if unit == "s" else "Peak memory (MB)"}}}}} \\']
        for m in cost.index:
            if m in NO_COST or m in LETTER_ONLY:
                continue
            vals = [cost_bb[bb].loc[m, f'{bus} {unit}'] for bus, bb in cols]
            tex.append(f'{LABEL[m]} & ' + ' & '.join('--' if np.isnan(v) else fmt(v) for v in vals) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (args.tex / 'table_cost_backbones.tex').write_text('\n'.join(tex) + '\n')   # for the letter

    # Summary for the text: signed gaps to FR; distances to Retrain_b, whose distance to FR is the band
    log('== per scenario: forgotten-station ROC; signed gap to FR; distance to Retrain_b: |ROC|, Fidelity, '
        'prob. diff, |MIA| (Fidelity to FR, the same seed, for comparison); retained-station ROC gap to FR')
    for bus, scen in scenarios(T):
        s = T[(T.feeder == bus) & (T.Scenario == scen)]
        fr = s[s.method == 'Retrain']
        log(f'== {bus} {scen}: FR forgotten-station ROC {fr.ROC_forget.mean():.3f} (sd {fr.ROC_forget.std():.3f}, '
            f'n {len(fr)}); band |ROC| {fr.dist_ROC_forget.mean():.4f}, Fidelity {fr.fidelity.mean():.3f}, '
            f'prob. diff {fr.probdiff.mean():.3f}, |MIA| {fr.dist_MIA_forget.mean():.3f}')
        for m in methods:
            g = s[s.method == m]
            if len(g) and m not in ('Retrain', 'Retrain_b'):
                log(f'   {m:12s} ROC {g.ROC_forget.mean():.3f}  signed gap {g.gap_ROC_forget.mean():+.4f}  '
                    f'|ROC| {g.dist_ROC_forget.mean():.4f}  Fidelity {g.fidelity.mean():.3f} '
                    f'(to FR {g.fidelity_FR.mean():.3f})  prob. diff {g.probdiff.mean():.3f}  '
                    f'|MIA| {g.dist_MIA_forget.mean():.3f}  retained ROC gap {g.gap_ROC_retain.mean():+.4f}')
    log('== per scenario, Table III: forgotten-station F1 at the threshold of 0.5, signed gap to FR (Wilcoxon over the '
        'trials, paired by backbone and seed, two-sided), and the share of the decisions at the forgotten stations '
        'that raise an alarm')
    for bus, scen in scenarios(T):
        s = T[(T.feeder == bus) & (T.Scenario == scen)]
        fr = s[s.method == 'Retrain']
        log(f'== {bus} {scen}: FR F1 {fr.F1_forget.mean():.3f} (sd {fr.F1_forget.std():.3f}), alarm {fr.alarm.mean():.3f}')
        for m in methods:
            g = s[s.method == m]
            if len(g) and m not in ('Retrain', 'Retrain_b'):
                log(f'   {m:12s} F1 {g.F1_forget.mean():.3f}  gap {g.gap_F1_forget.mean():+.4f} '
                    f'(p {wilcoxon_p(g.gap_F1_forget):.3g})  alarm {g.alarm.mean():.3f}')
    # |MIA-AUC - FR| per scenario for the letter (R1-3(iv)), against FR as in the paper, not the second FR;
    # Original is scored with the charging power and has no MIA
    log('== letter, R1-3(iv): forgotten-station MIA-AUC per scenario, the signed gap to FR (Wilcoxon over the trials, '
        'paired by backbone and seed, two-sided) and the mean absolute gap |MIA - FR|')
    for bus, scen in scenarios(T):
        s = T[(T.feeder == bus) & (T.Scenario == scen)]
        fr = s[s.method == 'Retrain']
        log(f'== {bus} {scen}: FR MIA {fr.MIA_forget.mean():.3f} (sd {fr.MIA_forget.std():.3f})')
        for m in methods:
            g = s[s.method == m]
            if len(g) and m not in ('Original', 'Retrain', 'Retrain_b'):
                log(f'   {m:12s} MIA {g.MIA_forget.mean():.3f}  gap {g.gap_MIA_forget.mean():+.4f} '
                    f'(p {wilcoxon_p(g.gap_MIA_forget):.3g})  |gap| {g.gap_MIA_forget.abs().mean():.4f}')
    log(f'== cost per request, S1, quiet phase, {COST_BACKBONE} as in the paper: median seconds / peak MB')
    for m, r in cost.iterrows():
        log(f'   {m:12s} ' + ', '.join(f'{b} {r[f"{b} s"]:.1f} s / {r[f"{b} MB"]:.0f} MB' for b in FEEDERS))
    log(f'== cost ratios, {COST_BACKBONE}: FT_e100 over Retrain in time ' + ', '.join(
        f'{b} {cost.loc["FT_e100", f"{b} s"] / cost.loc["Retrain", f"{b} s"]:.2f}' for b in FEEDERS)
        + '; GIF and IDEA over FT_e25 in memory ' + ', '.join(
        f'{b} {cost.loc[m, f"{b} MB"] / cost.loc["FT_e25", f"{b} MB"]:.2f}' for m in ('GIF', 'IDEA_cert') for b in FEEDERS)
        + '; FT_e25 over GIF in memory ' + ', '.join(
        f'{b} {cost.loc["FT_e25", f"{b} MB"] / cost.loc["GIF", f"{b} MB"]:.2f}' for b in FEEDERS))
    log('== letter: cost per request for every backbone, S1, quiet phase: median seconds / peak MB')
    for bb, c in cost_bb.items():
        for m, r in c.iterrows():
            log(f'   {bb} {m:12s} ' + ', '.join(f'{b} {r[f"{b} s"]:.1f} s / {r[f"{b} MB"]:.0f} MB'
                                               for b in FEEDERS if not np.isnan(r[f'{b} s'])))
    log(f'== yearly cost at 10^3 requests (one station each), {COST_BACKBONE}, hours: ' + '; '.join(
        f'{b}: ' + ', '.join(f'{m} {cost.loc[m, f"{b} s"] * 1e3 / 3600:.1f}'
                             for m in ('FT_e25', 'FT_e100', 'WarmStart', 'Retrain') if not np.isnan(cost.loc[m, f'{b} s']))
        for b in FEEDERS))

    # 6. Statistics against FR and against the band (R1-B3, R3-4): paired over the trials of a scenario
    S = paired_stats(T, [m for m in methods if m not in ('Original', 'Retrain', 'Retrain_b')])
    S.to_csv(args.out / 'table_stats.csv', index=False)
    log('== paired tests over the trials of a scenario (Wilcoxon signed-rank): signed gap to FR, two-sided; '
        '|ROC| distance to Retrain_b against that of FR, the band, one-sided (method farther); Fidelity to '
        'Retrain_b against that of FR, one-sided (method lower)')
    for _, r in S.iterrows():
        log(f'   {r.feeder} {r.Scenario} {r.method:12s} n {r.n:2d}  Macro ROC gap {r.macro_gap:+.4f} (p {r.p_macro:.3g})  '
            f'retained ROC gap {r.retained_gap:+.4f}  forgotten ROC gap {r.forget_gap:+.4f} (p {r.p_forget:.3g})  '
            f'|ROC| {r.dist_forget:.4f} vs band {r.band_dist:.4f} (p {r.p_band:.3g})  Fidelity {r.fidelity:.3f} vs '
            f'band {r.band_fidelity:.3f} (p {r.p_band_fidelity:.3g})')

    # 7. Epochs to early stopping (cap 3000)
    log('== epochs to early stopping, median (max): Original, Retrain, Retrain_b, FT to convergence; per scenario '
        'and over every scenario of a feeder (all)')
    for bus, scen in scenarios(T) + [(b, 'all') for b in FEEDERS]:
        s = T[(T.feeder == bus) & ((T.Scenario == scen) | (scen == 'all'))]
        e = {m: s[s.method == m].Epochs_run for m in ('Original', 'Retrain', 'Retrain_b', 'WarmStart')}
        ratio = (s[s.method == 'WarmStart'].set_index(KEY).Epochs_run / s[s.method == 'Retrain'].set_index(KEY).Epochs_run)
        log(f'   {bus} {scen}: ' + ', '.join(f'{m} {v.median():.0f} ({v.max():.0f})' for m, v in e.items())
            + f'; FT to convergence over Retrain: ratio of the medians {e["WarmStart"].median() / e["Retrain"].median():.2f}, '
            f'median of the paired ratios {ratio.median():.2f}')

    # 8. Component ablation for the letter (R1-B2, R3-1): the GDGU arms of E3
    A = ablation_trials()
    A.to_csv(args.out / 'ablation_trials.csv', index=False)
    tab = []
    for m, label in ABLATION:
        row = {'method': m}
        for bus in ('34bus', '123bus'):
            g = A[(A.feeder == bus) & (A.method == m)]
            for c in ('Macro_ROC', 'ROC_forget', 'MIA_forget'):
                row[f'{bus} {c}'] = g[f'gap_{c}'].mean()
        tab.append(row)
    tab = pd.DataFrame(tab).set_index('method')
    tab.to_csv(args.out / 'table_ablation.csv')
    cols = [(b, c) for b in ('34bus', '123bus') for c in ('Macro_ROC', 'ROC_forget', 'MIA_forget')]
    tex = [r'\begin{table}[t]', r'\centering',
           r'\caption{Component ablation (E3): gap to FR of the same trial, mean over the scenarios, '
           r'backbones and seeds 42 to 51. Utility is the Macro ROC-AUC over all stations.}', r'\label{tab:ablation}',
           r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
           mr('Arm') + r' & \multicolumn{3}{c}{34-bus} & \multicolumn{3}{c}{123-bus} \\',
           r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}', r' & Utility & ROC$_f$ & MIA$_f$ & Utility & ROC$_f$ & MIA$_f$ \\',
           r'\midrule']
    for m, label in ABLATION:
        tex.append(f'{label} & ' + ' & '.join(signed(tab.loc[m, f"{b} {c}"]) for b, c in cols) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (args.tex / 'table_ablation.tex').write_text('\n'.join(tex) + '\n')
    log('== ablation (E3): signed gap to FR, mean over scenarios, backbones and seeds (Macro ROC / forgotten ROC / MIA)')
    for m, label in ABLATION:
        log(f'   {m:15s} ' + ', '.join(f'{b} {tab.loc[m, f"{b} Macro_ROC"]:+.4f} / {tab.loc[m, f"{b} ROC_forget"]:+.4f} '
                                       f'/ {tab.loc[m, f"{b} MIA_forget"]:+.4f}' for b in ('34bus', '123bus')))
    for bus, need in (('34bus', 5), ('123bus', 8)):
        cells = preregistered(A[A.feeder == bus])
        log(f'   pre-registered rule (10/1), {bus}: GDGU_ls against BN + FT 25, paired by seed; a cell counts when '
            f'the Macro ROC gain is above 0.005 with one-sided p < 0.05: {cells.counts.sum()}/{len(cells)} cells '
            f'(needed {need}); mean gains {cells.gain.min():+.4f} to {cells.gain.max():+.4f}; '
            f'forgotten ROC {cells.gain_forget.min():+.4f} to {cells.gain_forget.max():+.4f}, '
            f'MIA {cells.gain_mia.min():+.4f} to {cells.gain_mia.max():+.4f}')
    # R3-1: the correction against FT alone and against a random step followed by the same BN and FT, cell by cell
    log('== letter, R3-1: GDGU against FT alone (FT 25, no correction, no BN) and against a random step + BN + FT 25 '
        '(E3), per backbone x scenario cell, paired by seed, two-sided Wilcoxon: range of the cell mean differences '
        'and cells with p < 0.05, Macro ROC (utility) / forgotten ROC / MIA')
    for bus in ('34bus', '123bus'):
        a = A[A.feeder == bus]
        e3 = e3_raw(bus)
        med = e3[e3.Method.isin(['GDGU_ls', 'GDGU_submitted', 'Random_step'])].groupby('Method').Step_len.median()
        log(f'   {bus}: median step length GDGU_ls {med["GDGU_ls"]:.3f}, GDGU_submitted {med["GDGU_submitted"]:.3f}, '
            f'Random_step {med["Random_step"]:.3f}')
        for arm in ('GDGU_ls', 'GDGU_submitted'):
            for ctrl in ('Finetune_pure', 'Random_step'):
                parts = []
                for c in ('Macro_ROC', 'ROC_forget', 'MIA_forget'):
                    cells = control_cells(a, arm, ctrl, c)
                    parts.append(f'{c} {cells["mean"].min():+.4f} to {cells["mean"].max():+.4f}, '
                                 f'p < 0.05 in {(cells.p < 0.05).sum()}/{len(cells)}')
                log(f'   {bus} {arm} vs {ctrl}: ' + '; '.join(parts))

    # 9. Sensitivity of the correction step to rho and lambda for the letter (R1-RA10, R2-7, R3-3)
    if SENS.exists():                       # the E5 runs of the letter, not needed for the paper
        V, Vt = sensitivity()
        V.to_csv(args.out / 'table_sensitivity.csv', index=False)
        Vt.to_csv(args.out / 'sensitivity_trials.csv', index=False)
        tex = [r'\begin{table}[t]', r'\centering',
               r'\caption{Sensitivity of the correction step (E5, 34-bus, S1, three backbones, seeds 42 to 46; every arm '
               r'is followed by BN recalibration and FT 25): median step length, share of requests where the clip acts, '
               r'gap to FR, and the Macro ROC-AUC difference to BN + FT 25 without a correction (paired, '
               r'Wilcoxon $p$).}', r'\label{tab:sensitivity}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
               r'\begin{tabular}{llrrrrrrr}', r'\toprule',
               ' & '.join(mr(h) for h in ('Rule', 'Value', 'Step', 'Clipped'))
               + r' & \multicolumn{3}{c}{Gap to FR} & \multicolumn{2}{c}{vs. BN + FT 25} \\',
               r'\cmidrule(lr){5-7} \cmidrule(lr){8-9}', r' & & & & Utility & ROC$_f$ & MIA$_f$ & Utility & $p$ \\',
               r'\midrule']
        for _, r in V.iterrows():
            val = f'{r.symbol} = {r.value}' if r.symbol else ''
            step = '' if np.isnan(r.step) else f'{r.step:.3f}'
            clip = '' if np.isnan(r.clipped) else f'{r.clipped:.0%}'.replace('%', r'\%')
            vs = '' if r.arm == 'Finetune_only' else signed(r.vs_bn, 4)
            p = '' if r.arm == 'Finetune_only' or np.isnan(r.p_vs_bn) else f'{r.p_vs_bn:.2g}'
            tex.append(f'{r.kind} & {val} & {step} & {clip} & {signed(r.macro_gap)} & {signed(r.forget_gap)} & '
                       f'{signed(r.mia_gap)} & {vs} & {p}' + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        (args.tex / 'table_sensitivity.tex').write_text('\n'.join(tex) + '\n')
        log('== sensitivity of the correction step (E5, 34-bus S1): median step, clipped share, gap to FR '
            '(Macro ROC / forgotten ROC / MIA), Macro ROC against BN + FT 25 (paired mean, Wilcoxon p)')
        for _, r in V.iterrows():
            log(f'   {r.arm:18s} n {r.n:2d}  step {r.step:.4f}  clipped {r.clipped:.2f}  gap {r.macro_gap:+.4f} / '
                f'{r.forget_gap:+.4f} / {r.mia_gap:+.4f}  vs BN + FT 25 {r.vs_bn:+.4f} (p {r.p_vs_bn:.3g})')

    # 10. E4 and E10a for the letter, from the tables of results/2026-10-15_E4E10/analysis_e4e10.py
    e4_e10_tables(args, log)
    # 11. Numbers of the letter outside the tables
    letter_numbers(log)
    # 12. Fidelity of the sequential runs and under noise, to Retrain_b
    agreement_e4_e9(log)
    (args.out / 'tables.txt').write_text('\n'.join(lines) + '\n')
    print(f'wrote {args.out} and {args.tex}')


if __name__ == '__main__':
    main()
