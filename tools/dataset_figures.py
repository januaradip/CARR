#!/usr/bin/env python
"""
Dataset statistics and protocol figures of the CARR article (Fig. 4 and Fig. 7) and the main counts
of Tables 1-3 and 6.

Usage:
  python tools/dataset_figures.py --data /path/to/CARR --out figures
"""
import argparse, csv, os, sys, collections as C, statistics as st
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from carr.common import Paths

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help='dataset root (release or packed layout)')
ap.add_argument('--out', default='figures')
ap.add_argument('--video-splits', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'splits', 'video_splits.csv'))
A = ap.parse_args()
OUT = A.out; os.makedirs(OUT, exist_ok=True)
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

# ---------------- data
PATHS = Paths(A.data, work=os.path.join(OUT, '.work'))
M = PATHS.manifest_rows()
read = PATHS.label_reader('labels')
box = {r['file_name']: tuple(map(float, read(r['file_name']).split()[1:5])) for r in M}
grp = lambda p: 'dis' if p in DISP else 'nondis'
vids = C.defaultdict(list)
for r in M: vids[r['video_id']].append(r)
vown = {v: C.Counter(x['participant_code'] for x in l).most_common(1)[0][0] for v, l in vids.items()}

# ================= Fig. 4: dataset statistics
fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.6, wspace=0.35)
# (a) frames per class by group
ax = fig.add_subplot(gs[0, 0]); clean(ax)
cnt = C.Counter((r['class_id'], grp(r['participant_code'])) for r in M)
y = np.arange(6)[::-1]
a = np.array([cnt[(str(i), 'nondis')] for i in range(6)]); b = np.array([cnt[(str(i), 'dis')] for i in range(6)])
ax.barh(y, a / 1e3, height=0.62, color=NONDIS, edgecolor=SURF, lw=1)
ax.barh(y, b / 1e3, left=a / 1e3, height=0.62, color=DIS, edgecolor=SURF, lw=1)
for yi, t in zip(y, a + b): ax.text(t / 1e3 + 1, yi, f'{t/1e3:.1f}k', va='center', fontsize=7, color=INK2)
ax.set_yticks(y, CLS); ax.set_xlabel('Frames (thousands)'); ax.set_xlim(0, 80)
ax.set_title('a  Frames per activity class')
# (b) frames per participant
ax = fig.add_subplot(gs[0, 1]); clean(ax, 'y')
pc = C.Counter(r['participant_code'] for r in M); P = sorted(pc)
ax.bar(range(len(P)), [pc[p] / 1e3 for p in P], width=0.66, color=[DIS if p in DISP else NONDIS for p in P])
ax.set_xticks(range(len(P)), [p[1:] for p in P], fontsize=7); ax.set_xlabel('Participant (P00–P14)')
ax.set_ylabel('Frames (thousands)'); ax.set_title('b  Frames per participant')
for i, p in enumerate(P):
    if p in AID: ax.text(i, pc[p] / 1e3 + 2, AID[p], ha='center', fontsize=6.5, color=INK2)
ax.text(0.98, 0.95, 'W wheelchair · C crutches · K walking cane', transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color=MUTED)
# (c) video duration
ax = fig.add_subplot(gs[1, 0]); clean(ax, 'y')
dur = {g: [len(l) / 15 for v, l in vids.items() if grp(vown[v]) == g] for g in ('nondis', 'dis')}
bins = np.arange(0, 105, 5)
ax.hist([np.clip(dur['nondis'], 0, 102), np.clip(dur['dis'], 0, 102)], bins=bins, color=[NONDIS, DIS], edgecolor=SURF, lw=0.6, stacked=True)
ax.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '≥100']); ax.set_xlabel('Annotated duration per video (s, 15 fps)'); ax.set_ylabel('Videos')
med = st.median([x for g in dur.values() for x in g]); ax.axvline(med, color=INK2, lw=0.8)
ax.text(med + 2, ax.get_ylim()[1] * 0.92, f'median {med:.1f} s', fontsize=7, color=INK2)
ax.set_title('c  Video length distribution')
# (d) participants per location
ax = fig.add_subplot(gs[1, 1]); clean(ax)
lf = C.defaultdict(C.Counter)
for r in M: lf[r['location_name']][grp(r['participant_code'])] += 1
L = sorted(lf, key=lambda l: lf[l]['nondis'] + lf[l]['dis'])
yy = np.arange(len(L))
an = np.array([lf[l]['nondis'] for l in L]) / 1e3; bd = np.array([lf[l]['dis'] for l in L]) / 1e3
ax.barh(yy, an, height=0.66, color=NONDIS, edgecolor=SURF, lw=1); ax.barh(yy, bd, left=an, height=0.66, color=DIS, edgecolor=SURF, lw=1)
ax.set_yticks(yy, [l.replace(' Store', '') for l in L], fontsize=6.5); ax.tick_params(axis='y', pad=2); ax.set_xlim(0, (an + bd).max() * 1.06); ax.set_xlabel('Frames (thousands)'); ax.set_title('d  Frames per recording location')
# (e) box centre heatmap
ax = fig.add_subplot(gs[2, 0])
xs = np.array([box[r['file_name']][0] for r in M]); ys = np.array([box[r['file_name']][1] for r in M])
h, _, _ = np.histogram2d(ys, xs, bins=[45, 80], range=[[0, 1], [0, 1]])
im = ax.imshow(np.log1p(h), extent=[0, 1, 1, 0], cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#104281']), aspect=9 / 16 * 1.0)
ax.set_aspect(9 / 16); ax.set_xticks([0, .5, 1], ['0', '0.5', '1']); ax.set_yticks([0, .5, 1], ['0', '0.5', '1']); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_color(GRID)
ax.set_xlabel('Normalised x'); ax.set_ylabel('Normalised y'); ax.set_title('e  Bounding-box centre density')
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04); cb.set_label('log(1 + frames)', color=INK2, fontsize=7); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6)
# (f) participant demographics
ax = fig.add_subplot(gs[2, 1]); clean(ax, 'both')
DEMO = {'P00': ('F', 20, 52, 172, '-'), 'P01': ('M', 22, 77, 175, '-'), 'P02': ('M', 21, 53, 180, '-'), 'P03': ('F', 21, 66, 158, '-'),
        'P04': ('F', 20, 65, 155, '-'), 'P05': ('F', 54, 68, 158, '-'), 'P06': ('M', 21, 50, 160, '-'), 'P07': ('M', 52, 102, 171, 'W'),
        'P08': ('M', 21, 115, 175, 'C'), 'P09': ('M', 21, 70, 160.5, 'K'), 'P10': ('M', 47, 79, 168, 'C'), 'P11': ('M', 22, 82, 175, 'W'),
        'P12': ('F', 21, 42, 155, 'K'), 'P13': ('M', 56, 52, 164, 'C'), 'P14': ('F', 59, 75, 158, '-')}
CL = sorted([p for p, d in DEMO.items() if d[1] <= 22 and d[2] < 100], key=lambda p: DEMO[p][2])
LY = {p: 40 + 6 * k for k, p in enumerate(CL)}
for p, (sx, age, wt, ht, aid) in DEMO.items():
    col = DIS if p in DISP else NONDIS
    ax.scatter(age, wt, s=46, marker='o' if sx == 'M' else 's', color=col, edgecolor='white', linewidth=1.2, zorder=3)
    lab = p[1:] + ('' if aid == '-' else f' {aid}')
    if p in LY:
        ax.annotate(lab, (age, wt), xytext=(27, LY[p]), fontsize=6, color=INK2, va='center', ha='left',
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=0, shrinkB=3))
    else:
        ax.text(age + 1.2, wt + 1.5, lab, fontsize=6, color=INK2, ha='left', va='center')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Body weight (kg)'); ax.set_xlim(17, 64); ax.set_ylim(35, 125)
ax.scatter([], [], marker='o', color=MUTED, s=30, label='male'); ax.scatter([], [], marker='s', color=MUTED, s=30, label='female')
ax.legend(loc='upper center', fontsize=6.5, handletextpad=0.2, ncol=2)
ax.set_title('f  Participant age and body weight')
fig.legend(handles=[Patch(color=NONDIS, label='Participants without disabilities'), Patch(color=DIS, label='Participants with disabilities')],
           loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.995), fontsize=8)
fig.subplots_adjust(top=0.92)
save(fig, 'Fig4_dataset_statistics')

# ================= Fig. 7: evaluation protocols
va = list(csv.DictReader(open(A.video_splits)))
part = sorted({r['participant_code'] for r in va}); locs = ['Alif Store', 'Budi Jaya Store', 'Hafidz 1 Store', 'Hafidz 2 Store', 'Hari Store', 'Haza Mart', 'Rehana Store', 'Senyum Store', 'Senyum 2 Store', 'Senyum 3 Store', 'UNEJ Mart', 'KPRI Kalimantan', 'KPRI Sumatera']
prot = ['CS1', 'CS2', 'CS3', 'CS4', 'CL1', 'CL2', 'CL3', 'CL4', 'CG']
def is_test(r, p):
    return r[p] == 'test'
TR, TE, NA = '#d7e6f8', '#eb6834', '#f4f3f1'
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={'width_ratios': [len(part), len(locs)], 'wspace': 0.08})
for ax, items, key, title in ((axs[0], part, 'participant_code', 'a  Participants'), (axs[1], locs, 'location_name', 'b  Recording locations')):
    Mx = np.zeros((len(prot), len(items)))
    for i, p in enumerate(prot):
        for j, it in enumerate(items):
            rs = [r for r in va if r[key] == it]
            t = sum(is_test(r, p) for r in rs)
            Mx[i, j] = 2 if t == len(rs) else (1.5 if t else 1)
    for i in range(len(prot)):
        for j in range(len(items)):
            v = Mx[i, j]
            col = TE if v == 2 else ('#f6b89b' if v == 1.5 else TR)
            ax.add_patch(FancyBboxPatch((j + 0.07, i + 0.07), 0.86, 0.86, boxstyle='round,pad=0,rounding_size=0.12', fc=col, ec='none'))
    ax.set_xlim(0, len(items)); ax.set_ylim(len(prot), 0)
    ax.set_xticks(np.arange(len(items)) + 0.5, [x.replace(' Store', '') for x in items], rotation=90 if key == 'location_name' else 0, fontsize=7)
    ax.set_yticks(np.arange(len(prot)) + 0.5, prot if key == 'participant_code' else [])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(title, pad=14)
    for y0 in (4, 8): ax.axhline(y0, color=INK2, lw=0.6)
    if key == 'participant_code':
        ax.set_xticklabels([x[1:] for x in items]); ax.set_xlabel('Participant (P00–P14)')
        j0 = items.index('P07'); ax.plot([j0 + 0.05, j0 + 6.95], [-0.25, -0.25], color=INK2, lw=1, clip_on=False)
        ax.text(j0 + 3.5, -0.4, 'with disabilities', ha='center', va='bottom', fontsize=6.5, color=INK2, clip_on=False)
    else:
        j0 = locs.index('UNEJ Mart'); ax.plot([j0 + 0.05, j0 + 2.95], [-0.25, -0.25], color=INK2, lw=1, clip_on=False)
        ax.text(j0 + 1.5, -0.4, 'disability-participant sites', ha='center', va='bottom', fontsize=6.5, color=INK2, clip_on=False)
fig.legend(handles=[Patch(color=TR, label='Training / validation'), Patch(color=TE, label='Test (entirely)'), Patch(color='#f6b89b', label='Test (partly)')],
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
save(fig, 'Fig7_evaluation_protocols')

# ================= main counts (Tables 1-3 and 6)
vf = C.Counter(r['video_id'] for r in M)
print(f'{len(M):,} frames, {len(vids)} videos, {len(set(vown.values()))} participants, {len({r["location_name"] for r in M})} stores')
print('Frames with other people visible:', sum(r['from_multi_person_frame'] == '1' for r in M))
print('Videos / frames per class:')
for i, c in enumerate(CLS):
    vs = {r['video_id'] for r in M if r['class_id'] == str(i)}
    n = sum(vf[v] for v in vs)
    print(f'  {c:<22} {len(vs):>4} {n:>8,} {100 * n / len(M):5.1f}%')
for g in ('nondis', 'dis'):
    vs = [v for v in vids if grp(vown[v]) == g]
    print(f'{g:<7}: {len(vs)} videos, {sum(vf[v] for v in vs):,} frames, median {st.median(vf[v] for v in vs):.0f} frames per video')
print(f'Figures written to {OUT}/')
