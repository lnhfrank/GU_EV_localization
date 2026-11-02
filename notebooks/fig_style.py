"""Shared figure style of the revision notebooks: sizes, widths, colors, method names and panel labels.

The revision notebooks (`Revision_Figures.ipynb`, `E9_Leakage_Topology.ipynb`, `Revision_Tables.ipynb`) call
`use()` once and take their widths, colors and names from here, so a method has one color and one name in every
figure of the paper and the response letter, and the name is the one the tables give it (`scripts/make_tables.py`).
"""
from contextlib import contextmanager
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Widths, inches: IEEE column and text width; letter-only figures stay inside the 7.09 in text of the A4 letter
SINGLE, DOUBLE, LETTER = 3.5, 7.16, 6.9

# Font sizes, points, at the printed size; titles, axis labels and legends as large as the panels allow
LABEL = 10                 # titles, axis labels, colorbar labels
TICK = 9                   # tick labels
LEGEND = 9                 # legends
SMALL = 8                  # dense annotations: bus names on the maps, point labels

# Font sizes of the figures in the paper, points as printed, so that the double-column map and the single-column
# figures read alike; see printed()
PRINT = {'LABEL': 7, 'TICK': 6, 'LEGEND': 6, 'SMALL': 6}
PAPER_SCALE = DOUBLE / SINGLE    # a figure drawn at the text width and shown one column wide

# Colors: no grays anywhere, including the grid, legend frames and the default color cycle
GRID = '#C6DBEF'
BAND = '#76BC79'           # FR and its reference band, green as in the submitted figures
BAND_ALPHA = 0.35
RANDOM = '#8C564B'         # the random-guess line, dotted
ERR = dict(capsize=2, elinewidth=0.8)    # error bars

RC = {
    'font.family': 'Times New Roman', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'font.size': TICK, 'axes.titlesize': LABEL, 'axes.labelsize': LABEL, 'figure.titlesize': LABEL,
    'figure.labelsize': LABEL, 'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
    'legend.fontsize': LEGEND, 'legend.title_fontsize': LEGEND, 'legend.frameon': False,
    'legend.edgecolor': GRID,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 4.5,
    'errorbar.capsize': ERR['capsize'],
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
    'axes.prop_cycle': cycler(color=['#1F5F8B', '#F7AC53', '#2CA6A4', '#7B61B3', '#EC6E66', '#52AADC',
                                     '#D95F9E', '#76BC79']),
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'savefig.dpi': 300,
    'pdf.fonttype': 42,    # embed TrueType fonts
}


def use():
    """Apply the shared rcParams; call once at the top of a notebook."""
    plt.rcParams.update(RC)


@contextmanager
def printed(scale=1.0):
    """Fonts that print at PRINT for a figure drawn `scale` times as wide as it is shown in the paper: 1 for a
    figure included at its own size, PAPER_SCALE for one drawn at the text width and shown one column wide.

    Sets the rcParams and the module's LABEL, TICK, LEGEND and SMALL for the figures drawn inside the block, so
    save and show the figure inside it.
    """
    global LABEL, TICK, LEGEND, SMALL
    saved = {'LABEL': LABEL, 'TICK': TICK, 'LEGEND': LEGEND, 'SMALL': SMALL}
    size = {k: v * scale for k, v in PRINT.items()}
    rc = {'font.size': size['TICK'], 'axes.titlesize': size['LABEL'], 'axes.labelsize': size['LABEL'],
          'figure.titlesize': size['LABEL'], 'figure.labelsize': size['LABEL'], 'xtick.labelsize': size['TICK'],
          'ytick.labelsize': size['TICK'], 'legend.fontsize': size['LEGEND'], 'legend.title_fontsize': size['LEGEND']}
    LABEL, TICK, LEGEND, SMALL = size['LABEL'], size['TICK'], size['LEGEND'], size['SMALL']
    try:
        with plt.rc_context(rc):
            yield
    finally:
        LABEL, TICK, LEGEND, SMALL = saved['LABEL'], saved['TICK'], saved['LEGEND'], saved['SMALL']


# Methods: names as in the tables of make_tables.py; colors and markers the same in every figure
NAME = {
    'Original': 'Original (with power)',
    'Mask_only': 'Original (power removed)',
    'GIF': 'GIF',
    'IDEA_cert': 'IDEA',               # only at the noise of its certification bound
    'GIF+FT': 'GIF + FT 25',
    'FT_e10': 'FT 10', 'FT_e25': 'FT 25', 'FT_e50': 'FT 50', 'FT_e100': 'FT 100',
    'WarmStart': 'FT to convergence',
    'Retrain': 'FR',
    'Retrain_b': 'FR, another seed',
    # GDGU arms: the ablation and the request-size runs, response letter only
    'Correction_only': 'Correction only',
    'Correction_ls': 'Correction, line search',
    'BN_only': 'BN recalibration only',
    'Random_ls': 'Random step, same length',
    'Random_step': 'Random step + BN + FT 25',
    'Finetune_only': 'BN + FT 25',
    'GDGU_submitted': 'GDGU as submitted',
    'GDGU_ls': 'Correction, line search + BN + FT 25',
}
COLOR = {
    'Original': '#F7AC53',             # orange
    'Mask_only': '#7B61B3',            # purple
    'GIF': '#EC6E66',                  # red
    'GIF+FT': '#A8322D',               # dark red
    'IDEA_cert': '#9C2E6B',            # plum
    'FT_e10': '#9ECAE1', 'FT_e25': '#52AADC', 'FT_e50': '#2E7FC0', 'FT_e100': '#1F5F8B',   # blue, darker with epochs
    'WarmStart': '#2CA6A4',            # teal
    'Retrain': BAND,                   # green
    'Retrain_b': '#3E8E41',            # dark green
    'Correction_ls': '#D4A017',        # gold
    'BN_only': '#9AA42B',              # olive
    'Random_ls': '#C2185B',            # crimson
    'GDGU_ls': '#E8743B',              # burnt orange
}
MARKER = {'Mask_only': 'o', 'Original': 'o', 'GIF': 's', 'IDEA_cert': 'v', 'WarmStart': 'D',
          'Correction_ls': 's', 'BN_only': 'v', 'FT_e25': 'D', 'FT_e100': 'P', 'GDGU_ls': '^', 'Random_ls': 'o'}
# Method keys of the result files that differ from the table keys
ALIAS = {'Original_power': 'Original', 'Finetune_pure': 'FT_e25', 'FTpure_e10': 'FT_e10', 'FTpure_e50': 'FT_e50',
         'FTpure_e100': 'FT_e100', 'GIF_FT': 'GIF+FT'}


def key(m):
    return ALIAS.get(m, m)


def name(m):
    """The name the tables give method m."""
    return NAME[key(m)]


def color(m):
    return COLOR[key(m)]


def marker(m):
    return MARKER.get(key(m), 'o')


# Other encodings shared between notebooks
FEEDER = {'34bus': 'IEEE 34-bus', '123bus': 'IEEE 123-bus', '8500bus': 'IEEE 8500-node'}
NOISE = [(None, 'No noise', '#1F5F8B'), (20, '20 dB', '#52AADC'), (10, '10 dB', '#2CA6A4'), (0, '0 dB', '#F7AC53')]
SCENARIO = {'S1': '#1F5F8B', 'S2': '#2CA6A4', 'S3': '#F7AC53', 'S4': '#7B61B3', 'S5': '#EC6E66'}   # also E4 steps
ATTACK_TYPE = {1: '#7B61B3', 2: '#EC6E66', 3: '#2CA6A4', 4: '#F7AC53'}
STATION = ['#7B61B3', '#EC6E66', '#2CA6A4', '#F7AC53', '#1F5F8B']          # EVCS#1 to EVCS#5
METRIC = {'Macro_ROC': '#1F5F8B', 'ROC_forget': '#F7AC53', 'MIA_forget': '#2CA6A4'}
CLASS = {1: '#EC6E66', 0: '#52AADC'}                                       # attacked, not attacked


def panel(k):
    """Label of the k-th panel, counted from 0: (a), (b), ..."""
    return f'({ascii_lowercase[k]})'


def label_panels(axes, start=0):
    """Put (a), (b), ... in front of the panel titles in reading order; a panel without a title gets the letter.

    The title text is changed in place, so its size, weight and padding stay as set.
    """
    for k, ax in enumerate(np.ravel(axes)):
        text = ax.title.get_text()
        ax.title.set_text(f'{panel(start + k)} {text}' if text else panel(start + k))


# Legends under the panels of the paper figures, compact and close to the axes
BELOW = dict(handlelength=1.3, handletextpad=0.4, columnspacing=0.8, borderpad=0, borderaxespad=0)


def legends_below(fig, rows, gap=0.6, row_gap=0.25, **kw):
    """One legend per list of handles in `rows`, centered under the panels: the first `gap` legend font sizes
    below the lowest tick or axis label, each next one `row_gap` below the one before. It measures the drawn
    figure, so call it after tight_layout.
    """
    kw = {**BELOW, **kw}
    em = plt.rcParams['legend.fontsize'] / 72 * fig.dpi          # one legend font size in display units
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    frames = [ax.get_position() for ax in fig.axes]
    x = (min(p.x0 for p in frames) + max(p.x1 for p in frames)) / 2
    top = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes) - gap * em
    legends = []
    for handles in rows:
        leg = fig.legend(handles=handles, ncol=len(handles), loc='upper center',
                         bbox_to_anchor=(x, inv.transform((0, top))[1]), **kw)
        fig.canvas.draw()
        top = leg.get_window_extent().y0 - row_gap * em
        legends.append(leg)
    return legends
