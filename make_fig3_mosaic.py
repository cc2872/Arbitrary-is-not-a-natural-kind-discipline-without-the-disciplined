"""
Fig 3 -- knockout matrix as a heat-mosaic.
3x3 of packed tiles. Rows = rules, cols = removals. Each tile is the raw run data:
a seed x update grid (thousands of points). Each ROW is drawn in its rule's identity
colour (poison red, matching blue, silly grey -- shared with Figs 2/5); within a tile
saturated = kept (gap high), washed-out = collapsed (gap ~0). Install phase left of the
switch line, ghost phase to the right.

Diagonal (own anchor removed) -> tile fades pale after the switch (framed in the rule colour).
Off-diagonal (a different support removed) -> tile stays saturated.
Silly row has no harm/matching anchor -> removing those is a no-op, so the rule HOLDS by
construction; those two tiles are drawn as held grey tiles tagged 'structural'.

Gap = opp_control - opp_marked (avoid: poison, silly)  |  opp_marked - opp_control (match)
control = the single neutral berry (index 2); matches the paper's endpoint measure.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import figstyle
figstyle.set_pub_style()
plt.rcParams['hatch.linewidth'] = 0.6   # fine structural hatch

WIN = (1000, 1600)      # ghost phase only: pure persistence after oversight lifts
SMOOTH = 15             # temporal smoothing (suppresses per-update striping in held tiles)
GR, GC = 18, 18         # tile pixel-grid resolution (rows x cols of discrete cells)
GRID_LW = 0.0           # no internal cell borders -- smooth gradient field
GRID_EC = "none"
MESH_ALPHA = 1.0        # true identity colours; the white trajectory + dark stroke stays the hero
OUTDIR = "new figures"

# structural cells (silly x harm, silly x matching): the rule HOLDS by construction --
# it has no such anchor, so removing it is a no-op. Drawn as a held teal tile + flat high
# line, tagged 'structural' so it reads as "holds (untested)", never as missing/failed.
STRUCT_VAL = 0.24     # structural fill: the 'kept' teal (held by construction)
STRUCT_LINE_Y = 0.80  # flat 'held-high' sparkline height

# the shared 'mosaic' survival colourmap (ALL tiles, one scale): kept = teal, collapsed = orange/maroon.
CMAP = LinearSegmentedColormap.from_list("collapse_keep", [
    "#6d1709", "#a5301a", "#cc5a29", "#e39a3c", "#f0d98f", "#f3efe2",
    "#a9d9c0", "#5bbfa0", "#2a9d8a", "#12836a", "#0a5f49"])
NORM = TwoSlopeNorm(vmin=-0.04, vcenter=0.10, vmax=0.28)
FRAME = "#5c1c0b"     # coral frame marks the diagonal (own anchor removed)

# (csv, condition, marked, control, kind, status)
CELLS = {
    ('poison','harm')     : ('poison_hazardoff.csv',    '0', 0, 2, 'avoid', 'fall'),
    ('poison','matching') : ('extinct3_cleanghost.csv', '0', 0, 2, 'avoid', 'hold'),
    ('poison','watching') : ('extinct3_cleanghost.csv', '0', 0, 2, 'avoid', 'hold'),
    ('matching','harm')     : ('coord_cleanghost_10.csv', '1', 1, 2, 'match', 'hold'),
    ('matching','matching') : ('coord_flat_10.csv',       '1', 1, 2, 'match', 'fall'),
    ('matching','watching') : ('coord_cleanghost_10.csv', '1', 1, 2, 'match', 'hold'),
    ('silly','harm')     : (None, None, 1, 2, 'avoid', 'structural'),
    ('silly','matching') : (None, None, 1, 2, 'avoid', 'structural'),
    ('silly','watching') : ('extinct3_cleanghost.csv', '1', 1, 2, 'avoid', 'fall'),
}
ROWS = ['poison','matching','silly']
COLS = ['harm','matching','watching']
ROW_LABEL = {'poison':'poison\n(harm)', 'matching':'matching\n(conformity)', 'silly':'silly\n(enforcement)'}
COL_LABEL = {'harm':'remove\nharm', 'matching':'remove\nmatching', 'watching':'remove\nwatching'}

def _mavg(a, w):
    if w <= 1 or a.size < w: return a
    return np.convolve(a, np.ones(w)/w, mode='same')

def box_matrix(path, cond, marked, control, kind):
    rows = [x for x in csv.DictReader(open(path)) if x['condition'] == str(cond)]
    seeds = sorted(set(int(x['seed']) for x in rows))
    def opp(rs, b):
        e = np.array([float(x[f'enc{b}']) for x in rs]); n = np.array([float(x[f'eat{b}']) for x in rs])
        return np.divide(n, e, out=np.zeros_like(e), where=e > 0)
    M, order = [], []
    for s in seeds:
        d = sorted((x for x in rows if int(x['seed']) == s and WIN[0] <= int(x['update']) < WIN[1]),
                   key=lambda x: int(x['update']))
        g = (opp(d, control) - opp(d, marked)) if kind == 'avoid' else (opp(d, marked) - opp(d, control))
        g = _mavg(g, SMOOTH)
        M.append(g); order.append(np.mean(g))            # sort seeds by ghost-phase level
    M = np.vstack(M)[np.argsort(order)[::-1]]            # brightest (kept) seed on top
    # -- pixelate: bin columns into GC cells (mean), interpolate seeds up to GR rows --
    binned = np.stack([seg.mean(1) for seg in np.array_split(M, GC, axis=1)], axis=1)  # (nseed, GC)
    xs, xt = np.linspace(0, 1, M.shape[0]), np.linspace(0, 1, GR)
    grid = np.vstack([np.interp(xt, xs, binned[:, c]) for c in range(GC)]).T           # (GR, GC)
    # cosmetic: smooth across columns (time) so held tiles read as a field, not stripes.
    # edge-pad (replicate) before convolving so tile borders don't wash out toward zero.
    w = 5; pad = w // 2; k = np.ones(w) / w
    grid = np.vstack([np.convolve(np.pad(grid[r], pad, mode='edge'), k, mode='valid')
                      for r in range(grid.shape[0])])
    return grid[::-1]                                    # brightest (kept) seed toward top

VLO, VHI = -0.04, 0.28    # gap range -> vertical position of the sparkline (matches colorbar bounds)

def tile_traj(path, cond, marked, control, kind):
    """Seed-mean adherence-gap trajectory over the ghost window, mapped to [0,1] tile height."""
    rows = [x for x in csv.DictReader(open(path)) if x['condition'] == str(cond)]
    seeds = sorted(set(int(x['seed']) for x in rows))
    def opp(rs, b):
        e = np.array([float(x[f'enc{b}']) for x in rs]); n = np.array([float(x[f'eat{b}']) for x in rs])
        return np.divide(n, e, out=np.zeros_like(e), where=e > 0)
    gs = []
    for s in seeds:
        d = sorted((x for x in rows if int(x['seed']) == s and WIN[0] <= int(x['update']) < WIN[1]),
                   key=lambda x: int(x['update']))
        g = (opp(d, control) - opp(d, marked)) if kind == 'avoid' else (opp(d, marked) - opp(d, control))
        gs.append(_mavg(g, SMOOTH))
    m = min(len(g) for g in gs)
    traj = np.vstack([g[:m] for g in gs]).mean(0)
    x = np.linspace(0, 1, m)
    y = np.clip((traj - VLO) / (VHI - VLO), 0.03, 0.97)
    return x, y

fig = plt.figure(figsize=(6.4, 6.2))
gs = GridSpec(3, 3, figure=fig, left=0.17, right=0.86, top=0.88, bottom=0.14,
              wspace=0.0, hspace=0.0)
im = None
for i, rule in enumerate(ROWS):
    for j, rem in enumerate(COLS):
        ax = fig.add_subplot(gs[i, j])
        path, cond, marked, control, kind, status = CELLS[(rule, rem)]
        edges_x = np.linspace(0, 1, GC + 1); edges_y = np.linspace(0, 1, GR + 1)
        if status == 'structural':
            # held-by-construction: uniform 'kept' teal field + flat high line, tagged
            grid = np.full((GR, GC), STRUCT_VAL)
            ax.pcolormesh(edges_x, edges_y, grid, cmap=CMAP, norm=NORM,
                          edgecolors=GRID_EC, linewidth=GRID_LW, alpha=MESH_ALPHA)
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, hatch='////',
                                   edgecolor=(1, 1, 1, 0.11), linewidth=0.0, zorder=3))
            y0 = STRUCT_LINE_Y
            ax.plot([0, 1], [y0, y0], color='white', lw=1.3, solid_capstyle='round', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.7, foreground=(0.10, 0.10, 0.10, 0.75))])
            ax.text(0.5, 0.12, 'structural', ha='center', va='center', transform=ax.transAxes,
                    fontsize=6, style='italic', color=(1, 1, 1, 0.85), zorder=9)
            for sp in ax.spines.values(): sp.set_visible(False)
        else:
            grid = box_matrix(path, cond, marked, control, kind)
            ax.pcolormesh(edges_x, edges_y, grid, cmap=CMAP, norm=NORM,
                          edgecolors=GRID_EC, linewidth=GRID_LW, alpha=MESH_ALPHA)
            # sparkline: seed-mean gap trajectory across the ghost phase, over the field
            xs, ys = tile_traj(path, cond, marked, control, kind)
            ax.plot(xs, ys, color='white', lw=1.3, solid_capstyle='round', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.7, foreground=(0.10, 0.10, 0.10, 0.78))])
            if status == 'fall':                          # coral frame marks the diagonal
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(FRAME); sp.set_linewidth(1.8); sp.set_zorder(10)
            else:
                for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel(ROW_LABEL[rule], fontsize=8.5, rotation=0, ha='right', va='center',
                          labelpad=12)
        if i == 0:
            ax.set_title(COL_LABEL[rem], fontsize=8.5, pad=6)

# colorbar -- SATURATION encodes survival (hue is per-row); shown in neutral grey so it
# reads as "darkness = kept, pale = collapsed" independent of which rule.
cax = fig.add_axes([0.875, 0.16, 0.020, 0.66])
sm = ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, ticks=[0.0, 0.11, 0.22])
cb.ax.set_yticklabels(['collapsed\n(0.00)', 'partial\n(0.11)', 'kept\n(0.22)'], fontsize=7.5)
cb.ax.tick_params(length=2.5, width=0.7)
cb.outline.set_linewidth(0.5)
cb.set_label('how much of the rule survives  (per-encounter gap)',
             fontsize=8, rotation=270, labelpad=20)

# within-tile horizontal scale: a representative arrow under the mosaic
# (left = the update oversight is removed, right = 600 updates later)
ax_bar = fig.add_axes([0.395, 0.075, 0.20, 0.001]); ax_bar.set_axis_off()
ax_bar.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.35",
                                mutation_scale=10))
fig.text(0.495, 0.045, "within each tile: 0 → 600 updates after oversight removed",
         ha="center", va="top", fontsize=7.5, color="0.35")

figstyle.save(fig, "fig3_mosaic", title="Knockout matrix")
