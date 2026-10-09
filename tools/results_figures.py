#!/usr/bin/env python
"""
Result tables and figures of the CARR article from a finished benchmark:
  results_tables.csv  - Tables 10-13 (mean ± standard deviation over folds)
  Fig8_baselines, Fig9_confusion_matrices, Fig10_participant_groups (.png and .pdf)

Usage:
  python tools/results_figures.py --work carr_work --out figures
(--work is the folder given to run_benchmark.py; it contains results/ and runs/.)
"""
import argparse, csv, glob, json, os, statistics as st
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True, help='work folder with results/ and runs/')
ap.add_argument('--out', default='figures')
A = ap.parse_args()
OUT, RES, RUNS = A.out, os.path.join(A.work, 'results'), os.path.join(A.work, 'runs')
os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
NONDIS, DIS = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.titlecolor': INK, 'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
CLS = ['No Interest', 'Viewing', 'Turning to Shelf', 'Touching', 'Picking and Returning', 'Picking and Putting']
DISP = {f'P{i:02d}' for i in range(7, 14)}
AID = {'P07': 'W', 'P08': 'C', 'P09': 'K', 'P10': 'C', 'P11': 'W', 'P12': 'K', 'P13': 'C'}
def clean(ax, grid='x'):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)
def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ================= Tables 10-13
def _load(f):
    p = f'{RES}/{f}_results.csv'
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
ALL = {k: _load(k) for k in ('detection', 'skeleton', 'video', 'reid')}
ALL['act'] = ALL['skeleton'] + ALL['video']


def R(kind, model, prot, key):
    v = [float(r[key]) for r in ALL[kind] if r['model'] == model and r['split'].rstrip('0123456789') == prot
         and r.get(key, '') not in ('', 'nan', 'NaN')]
    if not v:
        return '–'
    return f'{st.mean(v):.3f}' if len(v) == 1 else f'{st.mean(v):.3f} ± {st.stdev(v):.3f}'


ACT = [('LSTM', 'LSTM'), ('ST-GCN', 'ST-GCN'), ('CTR-GCN', 'CTR-GCN'), ('R2Plus1D-18', 'R(2+1)D-18'), ('VideoMAE-B', 'VideoMAE-B')]
rows = [['Table', 'Model', 'Protocol / group', 'Metric', 'Value']]
for m, n in ACT:
    for p in ('CS', 'CL', 'CG'):
        rows.append(['10', n, p, 'macro-F1 window', R('act', m, p, 'win_f1_all')])
        rows.append(['10', n, p, 'macro-F1 video', R('act', m, p, 'vid_f1_all')])
for m in ('YOLOv8m', 'YOLOv12m'):
    for p in ('CS', 'CL', 'CG'):
        for k, lab in (('mAP50_all', 'mAP@50'), ('mAP50_95_all', 'mAP@50:95'), ('precision_all', 'precision'), ('recall_all', 'recall')):
            rows.append(['11', m, p, lab, R('detection', m, p, k)])
for m, n in (('BoT-R50', 'BoT-ResNet50'), ('ViT-B16', 'ViT-B/16')):
    for k, lab in (('rank1_all', 'Rank-1'), ('rank5_all', 'Rank-5'), ('mAP_all', 'mAP'),
                   ('rank1_crossloc', 'Rank-1 cross-store'), ('mAP_crossloc', 'mAP cross-store')):
        rows.append(['12', n, 'ReID', lab, R('reid', m, 'ReID', k)])
for m in ('YOLOv8m', 'YOLOv12m'):
    rows += [['13', m, 'without disabilities (CS)', 'mAP@50', R('detection', m, 'CS', 'mAP50_nondis')],
             ['13', m, 'with disabilities (CS)', 'mAP@50', R('detection', m, 'CS', 'mAP50_dis')],
             ['13', m, 'with disabilities (CG)', 'mAP@50', R('detection', m, 'CG', 'mAP50_all')]]
for m, n in ACT:
    rows += [['13', n, 'without disabilities (CS)', 'macro-F1 window', R('act', m, 'CS', 'win_f1_nondis')],
             ['13', n, 'with disabilities (CS)', 'macro-F1 window', R('act', m, 'CS', 'win_f1_dis')],
             ['13', n, 'with disabilities (CG)', 'macro-F1 window', R('act', m, 'CG', 'win_f1_all')]]
for m, n in (('BoT-R50', 'BoT-ResNet50'), ('ViT-B16', 'ViT-B/16')):
    rows += [['13', n, 'without disabilities (ReID)', 'mAP', R('reid', m, 'ReID', 'mAP_nondis')],
             ['13', n, 'with disabilities (ReID)', 'mAP', R('reid', m, 'ReID', 'mAP_dis')]]
with open(f'{OUT}/results_tables.csv', 'w', newline='') as fh:
    csv.writer(fh).writerows(rows)
print(f'Tables 10-13 -> {OUT}/results_tables.csv')

# ================= Fig. 8 and Fig. 10
SK, VI, DE = ALL['skeleton'], ALL['video'], ALL['detection']
if SK and VI and DE:
    models = ['LSTM', 'ST-GCN', 'CTR-GCN', 'R2Plus1D-18', 'VideoMAE-B']
    mlabel = {'R2Plus1D-18': 'R(2+1)D-18'}
    ROWS = SK + VI
    def vals(rows, m, p, k):
        return [float(r[k]) for r in rows if r['model'] == m and r['split'].rstrip('0123456789') == p and r[k] not in ('', 'nan')]
    PC = {'CS': '#2a78d6', 'CL': '#1baf7a', 'CG': '#eb6834'}
    PL = {'CS': 'Cross-subject (CS, 4 folds)', 'CL': 'Cross-location (CL, 4 folds)', 'CG': 'Cross-group (CG, 1 split)'}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [5, 2], 'wspace': 0.25})
    for axx, rows, mods, key, ylab, title in ((ax, ROWS, models, 'win_f1_all', 'Macro-F1 (window level)', 'a  Activity recognition'),
                                              (bx, DE, ['YOLOv8m', 'YOLOv12m'], 'mAP50_all', 'mAP@50', 'b  Frame-level detection')):
        clean(axx, 'y')
        x = np.arange(len(mods)); bw = 0.25
        for k, p in enumerate(('CS', 'CL', 'CG')):
            xs = x + (k - 1) * (bw + 0.02)
            for xi, m in zip(xs, mods):
                v = vals(rows, m, p, key)
                mu = sum(v) / len(v)
                axx.bar(xi, mu, width=bw, color=PC[p], label=PL[p] if (xi == xs[0] and axx is ax) else None)
                if len(v) > 1:
                    axx.scatter([xi] * len(v), v, s=7, color=INK, zorder=3, linewidths=0)
        axx.set_xticks(x, [mlabel.get(m, m) for m in mods], fontsize=7); axx.set_ylabel(ylab); axx.set_title(title)
    ax.axhline(1 / 6, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.text(-0.45, 1 / 6 + 0.012, 'chance', fontsize=6.5, color=MUTED, bbox=dict(fc='white', ec='none', pad=0.3))
    ax.set_ylim(0, 0.75); bx.set_ylim(0, 0.75)
    ax.axvline(2.5, color=GRID, lw=1); ax.text(1, 0.71, 'Skeleton-based', ha='center', fontsize=7, color=INK2); ax.text(3.5, 0.71, 'Video-based', ha='center', fontsize=7, color=INK2)
    ax.scatter([], [], s=7, color=INK, label='individual fold')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=7)
    save(fig, 'Fig8_baselines')

    fig, ax = plt.subplots(figsize=(5.4, 2.6)); clean(ax, 'x')
    y = np.arange(len(models))[::-1]
    mean = lambda v: sum(v) / len(v)
    for yi, m in zip(y, models):
        nd, dd, cg = mean(vals(ROWS, m, 'CS', 'win_f1_nondis')), mean(vals(ROWS, m, 'CS', 'win_f1_dis')), mean(vals(ROWS, m, 'CG', 'win_f1_all'))
        ax.plot([min(cg, dd), nd], [yi, yi], color=GRID, lw=3, solid_capstyle='round', zorder=1)
        ax.scatter(nd, yi, s=44, color=NONDIS, edgecolor='white', linewidth=1.2, zorder=3)
        ax.scatter(dd, yi, s=44, color=DIS, edgecolor='white', linewidth=1.2, zorder=3)
        ax.scatter(cg, yi, s=40, color='white', edgecolor=DIS, linewidth=1.6, marker='D', zorder=3)
        ax.text(nd + 0.012, yi, f'Δ {nd - dd:.2f}', va='center', fontsize=6.8, color=INK2)
    ax.scatter([], [], s=44, color=NONDIS, label='Without disabilities (CS test)')
    ax.scatter([], [], s=44, color=DIS, label='With disabilities (CS test)')
    ax.scatter([], [], s=40, color='white', edgecolor=DIS, linewidth=1.6, marker='D', label='With disabilities, group unseen in training (CG)')
    ax.set_yticks(y, [mlabel.get(m, m) for m in models]); ax.set_xlabel('Macro-F1 (window level)'); ax.set_xlim(0.22, 0.66)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.22), ncol=1, fontsize=7)
    save(fig, 'Fig10_participant_groups')

    print('Fig. 8 and Fig. 10 written')

# ================= Fig. 9: confusion matrices
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'pdf.fonttype': 42, 'axes.titlesize': 8.5,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
CLS = ['No Interest', 'Viewing', 'Turning to Shelf', 'Touching', 'Picking and\nReturning', 'Picking and\nPutting']
SHORT = ['NI', 'V', 'TtS', 'T', 'P&R', 'P&P']
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
def cm(model, prot):
    pats = sorted(glob.glob(f'{RUNS}/*/*_{model}_{prot}*/confusion_window.json'))
    return sum(np.array(json.load(open(p)), float) for p in pats), len(pats)
panels = [('CTR-GCN', 'CS', 'a  CTR-GCN, cross-subject'), ('VideoMAE-B', 'CS', 'b  VideoMAE-B, cross-subject'),
          ('CTR-GCN', 'CG', 'c  CTR-GCN, cross-group'), ('VideoMAE-B', 'CG', 'd  VideoMAE-B, cross-group')]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6), gridspec_kw={'wspace': 0.08, 'hspace': 0.32})
for ax, (m, p, title) in zip(axs.flat, panels):
    M, n = cm(m, p)
    if n == 0:
        ax.set_visible(False); continue
    R = M / np.maximum(M.sum(1, keepdims=True), 1)
    im = ax.imshow(R, cmap=cmap, vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            v = R[i, j]
            ax.text(j, i, f'{v*100:.0f}', ha='center', va='center', fontsize=7, color='white' if v > 0.55 else '#0b0b0b')
    ax.set_xticks(range(6), SHORT); ax.set_yticks(range(6), SHORT if ax in axs[:, 0] else [''] * 6)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title + (f' ({n} folds)' if n > 1 else ''), pad=6)
    ax.set_xlabel('Predicted class')
    if ax in axs[:, 0]: ax.set_ylabel('True class')
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label('Proportion of true-class windows (%)')
cb.set_ticks([0, .25, .5, .75, 1], labels=['0', '25', '50', '75', '100']); cb.outline.set_visible(False)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/Fig9_confusion_matrices.{ext}', dpi=400, bbox_inches='tight', facecolor='white')
print('Fig. 9 written')
