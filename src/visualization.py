"""Visualization functions for GDGU/GIF/IDEA localization experiments.

All style constants are grouped at the top for easy customization.
Call `apply_style()` once before plotting to set Times New Roman globally.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
#  Style defaults — override via apply_style(overrides={...})
# ============================================================
STYLE = {
    # Font
    'font_family':   'Times New Roman',
    'fs_label':      26,
    'fs_tick':       24,
    'fs_legend':     20,
    'fs_subtitle':   22,
    'fs_annotation': 18,

    # Colors — 5 methods
    'colors': {
        'Original': '#C7CADE',
        'GDGU':     '#F7AC53',
        'GIF':      '#52AADC',
        'IDEA':     '#EC6E66',
        'Retrain':  '#76BC79',
    },
    'markers': {
        'Original': 'o',
        'GDGU':     's',
        'GIF':      'D',
        'IDEA':     'P',
        'Retrain':  '^',
    },
    'ideal_line_color': 'red',
    'grid_alpha':       0.3,
    'bar_alpha':        0.85,
    'bar_edge_color':   'black',
    'bar_edge_width':   0.5,
    'fill_alpha':       0.15,

    # Save
    'save_fmt': 'pdf',
    'save_dpi': 300,
}

METHODS_ORDER = ['Original', 'GDGU', 'GIF', 'IDEA', 'Retrain']
GU_METHODS = ['GDGU', 'GIF', 'IDEA']


def apply_style(overrides=None):
    """Apply matplotlib rcParams for Times New Roman. Call once before plotting.

    Args:
        overrides: dict to merge into STYLE, e.g. {'fs_label': 24, 'save_fmt': 'png'}.
    """
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(STYLE.get(k), dict):
                STYLE[k].update(v)
            else:
                STYLE[k] = v

    font = STYLE['font_family']
    mpl.rcParams.update({
        'font.family':      'serif',
        'font.serif':       [font],
        'mathtext.fontset':  'custom',
        'mathtext.rm':       font,
        'mathtext.it':       f'{font}:italic',
        'mathtext.bf':       f'{font}:bold',
        'axes.titlepad':     12,   # extra space below the GAT/GCN/GIN subplot titles
    })


# ============================================================
#  Data loading
# ============================================================
def _scenario_sort_key(s):
    """Sort key for scenario names like 'S1', 'S2-0', 'S3-1'.

    Returns (n, k) tuple so S1-0 < S1-1 < S1-2 < S2-0 < ...
    Legacy names without '-k' suffix get k=-1 so they sort first.
    """
    if '-' in s:
        parts = s.split('-')
        return (int(parts[0][1:]), int(parts[1]))
    return (int(s[1:]), -1)


def load_results(results_dir, bus_system='123bus', tag=None):
    """Load raw CSV from a date-named results folder.

    Args:
        results_dir: path to date folder, e.g. '.../results/2026-04-07'.
        bus_system:  '34bus' or '123bus'.
        tag:         Optional filename tag appended to bus_system.  For
                     example tag='raw' loads '{bus_system}_raw_results_raw.csv'.
                     Default None loads '{bus_system}_results_raw.csv'.

    Returns:
        df, scenarios, backbones
    """
    prefix = bus_system if tag is None else f'{bus_system}_{tag}'
    csv_path = os.path.join(results_dir, f'{prefix}_raw.csv')
    df = pd.read_csv(csv_path)

    scen_keys = sorted(df.Scenario.unique(), key=_scenario_sort_key)
    scenarios = {sk: {'label': sk} for sk in scen_keys}
    backbones = sorted(df.Backbone.unique())

    print(f'Loaded {len(df)} rows from {csv_path}')
    print(f'  Backbones : {backbones}')
    print(f'  Scenarios : {scen_keys}')
    print(f'  Methods   : {sorted(df.Method.unique())}')
    return df, scenarios, backbones


# ============================================================
#  Internal helpers
# ============================================================
def _savefig(fig, filepath):
    from matplotlib.transforms import Bbox
    # Asymmetric margins (inches): more on top (above GAT/GCN/GIN titles),
    # less on bottom (below the S1--S5 tick labels).
    pad_top, pad_bottom, pad_side = 0.45, 0.05, 0.15
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    bb = Bbox.from_extents(bb.x0 - pad_side, bb.y0 - pad_bottom,
                           bb.x1 + pad_side, bb.y1 + pad_top)
    fig.savefig(filepath, bbox_inches=bb, dpi=STYLE['save_dpi'])


def _scen_labels(scenarios):
    """Return short display labels for scenarios.

    For Sn-k format (e.g. 'S1-0'), use the key directly.
    For legacy format with full label, take the part before ':'.
    """
    labels = []
    for s in scenarios:
        lbl = scenarios[s]['label']
        labels.append(lbl.split(':')[0] if ':' in lbl else s)
    return labels


def _available_methods(df):
    """Return methods present in df, in METHODS_ORDER order."""
    present = set(df.Method.unique())
    return [m for m in METHODS_ORDER if m in present]


# ============================================================
#  Plot functions
# ============================================================
def plot_metric_bars(df, metric_col, ylabel, ylim, filepath,
                     scenarios, backbones):
    """Grouped bar chart (1x3 subplots, one per backbone)."""
    S = STYLE
    methods = _available_methods(df)
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(scen_keys))
        for m_idx, method in enumerate(methods):
            offset = (m_idx - (n_methods - 1) / 2) * width
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub[metric_col].mean())
                stds.append(sub[metric_col].std())
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=S['colors'][method], alpha=S['bar_alpha'],
                   capsize=3, edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=S['fs_label'])
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_metric_lines(df, metric_col, ylabel, ylim, filepath,
                      scenarios, backbones, xlabel='Scenario'):
    """Line chart with shaded std (1x3 subplots, one per backbone).

    Pass xlabel='' (or None) to omit the x-axis label.
    """
    S = STYLE
    methods = _available_methods(df)
    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    x = np.arange(len(scen_keys))
    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        for method in methods:
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub[metric_col].mean())
                stds.append(sub[metric_col].std())
            means, stds = np.array(means), np.array(stds)
            ax.plot(x, means, marker=S['markers'][method], color=S['colors'][method],
                    label=method, linewidth=2, markersize=8)
            ax.fill_between(x, means - stds, means + stds,
                            color=S['colors'][method], alpha=S['fill_alpha'])
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=S['fs_label'])
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=S['fs_label'])
        ax.set_ylim(ylim)
        ax.grid(True, alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_per_evcs_roc(df, filepath, scenarios, backbones, evcs_names=None):
    """backbone (row) x EVCS (col) grid, grouped bars per scenario."""
    S = STYLE
    methods = _available_methods(df)
    n_methods = len(methods)
    width = 0.8 / n_methods

    roc_cols = sorted([c for c in df.columns if c.startswith('ROC_EVCS')])
    n_evcs = len(roc_cols)
    n_bb = len(backbones)

    if evcs_names is None:
        evcs_names = [f'EVCS {i+1}' for i in range(n_evcs)]

    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    fig, axes = plt.subplots(n_bb, n_evcs, figsize=(6 * n_evcs, 5 * n_bb), sharey=True)
    if n_bb == 1:
        axes = [axes]
    if n_evcs == 1:
        axes = [[ax] for ax in axes]

    for row, bb in enumerate(backbones):
        for col, (roc_col, evcs_name) in enumerate(zip(roc_cols, evcs_names)):
            ax = axes[row][col]
            x = np.arange(len(scen_keys))
            for m_idx, method in enumerate(methods):
                offset = (m_idx - (n_methods - 1) / 2) * width
                means, stds = [], []
                for sk in scen_keys:
                    sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                    means.append(sub[roc_col].mean())
                    stds.append(sub[roc_col].std())
                ax.bar(x + offset, means, width, yerr=stds,
                       label=method, color=S['colors'][method], alpha=S['bar_alpha'],
                       capsize=3, edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
            if row == 0:
                ax.set_title(evcs_name, fontsize=S['fs_subtitle'] - 2, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=S['fs_tick'] - 2)
            ax.tick_params(axis='y', labelsize=S['fs_tick'] - 2)
            if col == 0:
                ax.set_ylabel(f'{bb}\nROC-AUC', fontsize=S['fs_label'] - 2)
            ax.set_ylim(0.4, 1.0)
            ax.grid(axis='y', alpha=S['grid_alpha'])
            if row == 0 and col == n_evcs - 1:
                ax.legend(fontsize=S['fs_legend'] - 2)
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_mia_metric(df, filepath, scenarios, backbones,
                    metric_col='MIA_AUC', ylabel='MIA-AUC'):
    """Bar chart for one MIA column (MIA_forget / MIA_retain / MIA_AUC).

    MIA_forget: approaches 0.5 means forgotten samples no longer recognizable.
    MIA_retain: should stay high, signalling retained memory on non-forget EVCS.
    MIA_AUC:    overall aggregate (less informative than forget-restricted).
    """
    S = STYLE
    mia_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                   if m in df.Method.unique()]
    n_methods = len(mia_methods)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    if len(backbones) == 1:
        axes = [axes]
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(scen_keys))
        for m_idx, method in enumerate(mia_methods):
            offset = (m_idx - (n_methods - 1) / 2) * width
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub[metric_col].mean())
                stds.append(sub[metric_col].std())
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=S['colors'].get(method, '#888'),
                   alpha=S['bar_alpha'], capsize=3,
                   edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.axhline(y=0.5, color=S['ideal_line_color'], linestyle='--', alpha=0.7,
                   label='Ideal (0.5)')
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=S['fs_label'])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_mia_auc(df, filepath, scenarios, backbones):
    """Backward-compatible wrapper: plots MIA_AUC (overall)."""
    plot_mia_metric(df, filepath, scenarios, backbones,
                    metric_col='MIA_AUC', ylabel='MIA-AUC (overall)')


def plot_mia_forget(df, filepath, scenarios, backbones):
    """MIA on forget EVCS labels only.  Closer to 0.5 = stronger erasure."""
    plot_mia_metric(df, filepath, scenarios, backbones,
                    metric_col='MIA_forget', ylabel='MIA-AUC (forget)')


def plot_mia_retain(df, filepath, scenarios, backbones):
    """MIA on retained EVCS labels.  Should stay high (utility preserved)."""
    plot_mia_metric(df, filepath, scenarios, backbones,
                    metric_col='MIA_retain', ylabel='MIA-AUC (retain)')


def plot_time_comparison(df, filepath, scenarios, backbones):
    """Time bar chart for all GU methods + Retrain, with speedup annotations."""
    S = STYLE
    gu_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                  if m in df.Method.unique()]
    n_methods = len(gu_methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(8.8, 5))
    scen_keys = list(scenarios.keys())

    n_scen = len(scen_keys)
    groups = [sk for bb in backbones for sk in scen_keys]   # scenario-only tick labels

    x = np.arange(len(groups))
    for m_idx, method in enumerate(gu_methods):
        offset = (m_idx - (n_methods - 1) / 2) * width
        means = []
        for bb in backbones:
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['Time'].mean())
        ax.bar(x + offset, means, width,
               label=method, color=S['colors'][method],
               edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=S['fs_tick'])   # S1--S5 directly under the axis
    trans = ax.get_xaxis_transform()
    for gi, bb in enumerate(backbones):
        left, right = gi * n_scen, gi * n_scen + (n_scen - 1)
        center = (left + right) / 2
        ax.plot([left - 0.4, right + 0.4], [-0.11, -0.11], transform=trans,
                color='black', lw=1.0, clip_on=False)              # group bracket
        ax.text(center, -0.145, bb, transform=trans, ha='center', va='top',
                fontsize=S['fs_tick'], fontweight='bold')           # GNN name (outer tier)
    ax.tick_params(axis='y', labelsize=S['fs_tick'])
    ax.set_ylabel('Time (seconds)', fontsize=S['fs_label'])
    ax.legend(fontsize=S['fs_legend'])
    ax.grid(axis='y', alpha=S['grid_alpha'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_memory_usage(df, filepath, scenarios, backbones):
    """Peak GPU memory bar chart for all GU methods + Retrain, single panel.

    Same layout as plot_time_comparison: x-axis groups by backbone and scenario.
    """
    S = STYLE
    if 'Peak_Memory_MB' not in df.columns:
        print('  [skip] Peak_Memory_MB column not found — skipping memory plot')
        return

    gu_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                  if m in df.Method.unique()]
    n_methods = len(gu_methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(8.8, 5))
    scen_keys = list(scenarios.keys())

    n_scen = len(scen_keys)
    groups = [sk for bb in backbones for sk in scen_keys]   # scenario-only tick labels

    x = np.arange(len(groups))
    for m_idx, method in enumerate(gu_methods):
        offset = (m_idx - (n_methods - 1) / 2) * width
        means = []
        for bb in backbones:
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['Peak_Memory_MB'].mean())
        ax.bar(x + offset, means, width,
               label=method, color=S['colors'][method],
               edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=S['fs_tick'])   # S1--S5 directly under the axis
    trans = ax.get_xaxis_transform()
    for gi, bb in enumerate(backbones):
        left, right = gi * n_scen, gi * n_scen + (n_scen - 1)
        center = (left + right) / 2
        ax.plot([left - 0.4, right + 0.4], [-0.11, -0.11], transform=trans,
                color='black', lw=1.0, clip_on=False)              # group bracket
        ax.text(center, -0.145, bb, transform=trans, ha='center', va='top',
                fontsize=S['fs_tick'], fontweight='bold')           # GNN name (outer tier)
    ax.tick_params(axis='y', labelsize=S['fs_tick'])
    ax.set_ylabel('Peak Memory (MB)', fontsize=S['fs_label'])
    ax.legend(fontsize=S['fs_legend'])
    ax.grid(axis='y', alpha=S['grid_alpha'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def _scenario_gradient(base_hex, n, lo=0.55, hi=1.0):
    """Return *n* RGBA colours ramping from light (lo) to full saturation (hi).

    Works by interpolating between white and the base colour in RGB space.
    S1 is lightest, S{n} is darkest — visually encodes scenario severity.
    """
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    alphas = np.linspace(lo, hi, n)
    return [(1 - a) * white + a * base for a in alphas]


def plot_f1_vs_mia(df, filepath, scenarios, backbones):
    """Forgetting-Reasoning trade-off scatter: Macro F1 (y) vs MIA-AUC (x).

    One subplot per backbone. Points connected by polyline S1→…→S{n} per
    method, with colour gradient (light S1 → dark S{n}). Only S1 and S{n}
    are labelled to avoid clutter.
    """
    S = STYLE
    mia_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                   if m in df.Method.unique()]
    scen_keys = list(scenarios.keys())
    n_scen = len(scen_keys)

    fig, axes = plt.subplots(1, len(backbones), figsize=(6 * len(backbones), 5.5),
                             sharey=True, sharex=True)
    if len(backbones) == 1:
        axes = [axes]

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        all_end_pts = []
        all_scatter_pts = []
        for method in mia_methods:
            grad = _scenario_gradient(S['colors'][method], n_scen)
            mia_means, f1_means = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                mia_means.append(sub['MIA_forget'].mean())
                f1_means.append(sub['Macro_F1'].mean())

            ax.plot(mia_means, f1_means, color=S['colors'][method],
                    alpha=0.35, linewidth=1.5, zorder=2)

            for i, sk in enumerate(scen_keys):
                label = method if i == n_scen - 1 else None
                ax.scatter(mia_means[i], f1_means[i],
                           color=grad[i], marker=S['markers'][method],
                           s=120, edgecolors='black', linewidths=0.6,
                           label=label, zorder=3)

            all_end_pts.append((method, mia_means[-1], f1_means[-1]))
            all_scatter_pts.extend(zip(mia_means, f1_means))

        sn_label = scen_keys[-1]
        px = 18
        # Only below-placements allowed: (dx, dy, ha, va). Keep labels close.
        candidates = [
            ( 0,   -px,       'center', 'top'),   # directly below
            ( px,  -px*0.7,   'left',   'top'),   # lower-right (small)
            (-px,  -px*0.7,   'right',  'top'),   # lower-left  (small)
            ( 0,   -px*1.9,   'center', 'top'),   # further directly below
        ]
        fig.canvas.draw()
        trans = ax.transData
        inv = trans.inverted()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scatter_px = [trans.transform((sx, sy)) for sx, sy in all_scatter_pts]
        placed_labels = []
        # Approximate label half-width in pixels for bounds check
        label_half_w = 12
        for method, mx, my in all_end_pts:
            anchor = trans.transform((mx, my))
            best_cand, best_score = None, -1e9
            for idx, cand in enumerate(candidates):
                dx, dy, ha, va = cand
                tx, ty = anchor[0] + dx, anchor[1] + dy
                # Convert candidate label position back to data coords and
                # check it fits inside the axis bounds (with label width).
                left_px  = tx - (label_half_w if ha == 'left' else
                                 -label_half_w if ha == 'right' else label_half_w)
                right_px = tx + (label_half_w if ha == 'left' else
                                 -label_half_w if ha == 'right' else label_half_w)
                lx, _ = inv.transform((min(left_px, right_px), ty))
                rx, _ = inv.transform((max(left_px, right_px), ty))
                _, by = inv.transform((tx, ty))
                if lx < xlim[0] or rx > xlim[1] or by < ylim[0]:
                    continue
                obstacles = scatter_px + placed_labels
                min_d = min((((tx - qx)**2 + (ty - qy)**2)**0.5
                             for qx, qy in obstacles), default=999)
                preference_bonus = (len(candidates) - idx) * 4
                score = min_d + preference_bonus
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand is None:
                best_cand = candidates[0]
            dx, dy, ha, va = best_cand
            ax.annotate(sn_label, (mx, my),
                        textcoords='offset points', xytext=(dx, dy),
                        fontsize=S['fs_annotation'], fontweight='bold',
                        color=S['colors'][method], ha=ha, va=va)
            placed_labels.append((anchor[0] + dx, anchor[1] + dy))

        ax.axvline(x=0.5, color=S['ideal_line_color'], linestyle='--', alpha=0.6)
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xlabel('MIA-AUC', fontsize=S['fs_label'])
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        ax.tick_params(axis='both', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel('F1 Score', fontsize=S['fs_label'])
        ax.grid(True, alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'], loc='best')
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_gu_comparison(df, filepath, scenarios, backbones):
    """GU method comparison: GDGU vs GIF vs IDEA across scenarios.

    One subplot per backbone. Lines for each GU method + dashed Retrain
    as gold-standard reference. Y-axis = Macro ROC-AUC.
    """
    S = STYLE
    gu_plus = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
               if m in df.Method.unique()]
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    x = np.arange(len(scen_keys))

    fig, axes = plt.subplots(1, len(backbones), figsize=(6 * len(backbones), 5.5),
                             sharey=True)
    if len(backbones) == 1:
        axes = [axes]

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        for method in gu_plus:
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['Macro_ROC'].mean())
                stds.append(sub['Macro_ROC'].std())
            means, stds = np.array(means), np.array(stds)
            ls = '--' if method == 'Retrain' else '-'
            lw = 1.5 if method == 'Retrain' else 2.5
            ax.plot(x, means, marker=S['markers'][method], color=S['colors'][method],
                    label=method, linewidth=lw, linestyle=ls, markersize=9)
            ax.fill_between(x, means - stds, means + stds,
                            color=S['colors'][method], alpha=S['fill_alpha'])

        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        ax.set_xlabel('Scenario', fontsize=S['fs_label'])
        if ax_idx == 0:
            ax.set_ylabel('Macro ROC-AUC', fontsize=S['fs_label'])
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'], loc='best')
    # plt.suptitle('GU Method Comparison', fontsize=S['fs_label'] + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


# ============================================================
#  Aggregate figure generation
# ============================================================
def plot_all_v6(df, output_dir, scenarios, backbones, bus_system='34bus'):
    """Generate all figures: utility (ExMatch / ROC-AUC / F1), MIA privacy,
    and efficiency (time / memory)."""
    os.makedirs(output_dir, exist_ok=True)
    fmt = STYLE['save_fmt']
    j = lambda name: os.path.join(output_dir, f'{bus_system}_{name}.{fmt}')

    # Utility plots
    plot_metric_bars(df, 'ExMatch', 'Exact Match Accuracy', (0.0, 1.0),
                     j('ExMatch_comparison'), scenarios, backbones)
    plot_metric_bars(df, 'Macro_ROC', 'Macro ROC-AUC', (0.4, 1.0),
                     j('MacroROC_comparison'), scenarios, backbones)
    plot_metric_bars(df, 'Macro_F1', 'Macro F1', (0.0, 1.0),
                     j('MacroF1_comparison'), scenarios, backbones)
    plot_metric_lines(df, 'Macro_ROC', 'Macro ROC-AUC', (0.4, 1.0),
                      j('MacroROC_trend'), scenarios, backbones)
    plot_per_evcs_roc(df, j('PerEVCS_ROC_breakdown'), scenarios, backbones)
    plot_mia_auc(df, j('MIA_AUC_comparison'), scenarios, backbones)
    plot_mia_forget(df, j('MIA_forget_comparison'), scenarios, backbones)
    plot_mia_retain(df, j('MIA_retain_comparison'), scenarios, backbones)
    plot_f1_vs_mia(df, j('F1_vs_MIA_tradeoff'), scenarios, backbones)
    plot_gu_comparison(df, j('GU_method_comparison'), scenarios, backbones)
    plot_time_comparison(df, j('Time_comparison'), scenarios, backbones)
    plot_memory_usage(df, j('Memory_usage'), scenarios, backbones)

    print(f"\nAll figures saved to {output_dir}/ (prefix: {bus_system}_)")
