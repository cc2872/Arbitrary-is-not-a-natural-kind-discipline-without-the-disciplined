"""Fig 5 -- The persistence-mechanism 2x2 (what the storage probes bought us).
Rows: environmental (poison) / coordination. Columns: cue present (frozen weights) /
cue absent (mark cue masked). Each cell = adherence retained through the ghost phase,
as a fraction of the pre-removal (normal) persistence, with per-seed points.

Reads at a glance: the two grounded rules persist by DIFFERENT means --
  poison    : frozen-stable (103%) but cue-DEPENDENT (drops to 46% when the mark is masked)
  coordinat.: cue-INDEPENDENT (103% masked) but only partly frozen-stable (78%),
              i.e. re-adaptation-sustained rather than purely weight-stored.
New plot, from the four-arm probe data (probe_{poison,coord}_{frozen,cuemask}).
  python make_fig5_storage.py
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle
figstyle.set_pub_style()

def perseed(path, cond, mi, nbt, mode, lo=1500, hi=1600):
    r=[x for x in csv.DictReader(open(path)) if x['condition']==cond]
    S=sorted(set(int(x['seed']) for x in r)); ctrl=max(t for t in range(nbt) if t!=mi)
    def o(s,i):
        rs=[x for x in r if int(x['seed'])==s and lo<=int(x['update'])<hi]
        e=sum(float(x[f'eat{i}']) for x in rs); c=sum(float(x[f'enc{i}']) for x in rs); return e/max(c,1e-9)
    return [ (o(s,ctrl)-o(s,mi)) if mode=='avoid' else (o(s,mi)-o(s,ctrl)) for s in S]

NORMAL = {"poison": 0.234, "coord": 0.203}   # pre-removal (normal) persistence
# (row, col) -> (csv | None, cond, mi, mode) ; None csv => summary-only cell
CELLS = {
 ("poison","frozen"):  ("probe_poison_frozen.csv","0",0,"avoid"),
 ("poison","cuemask"): ("probe_poison_cuemask.csv","0",0,"avoid"),
 ("coord","frozen"):   (None, None, None, None),      # summary only: +0.158 sd .005 n5
 ("coord","cuemask"):  ("probe_coord_cuemask.csv","1",1,"converge"),
}
COORD_FROZEN = dict(mean=0.158, sd=0.005, n=5)          # reported; per-seed CSV pending
ROWS = [("poison","Environmental\n(poison)",figstyle.PALETTE["environmental"]),
        ("coord","Coordination",figstyle.PALETTE["coordination"])]
COLS = [("frozen","Cue present\n(frozen weights)"), ("cuemask","Cue absent\n(mark cue masked)")]

def cell_data(row, col):
    csvf, cond, mi, mode = CELLS[(row, col)]
    norm = NORMAL[row]
    if csvf is None:
        return None, COORD_FROZEN["mean"], COORD_FROZEN["sd"], COORD_FROZEN["mean"]/norm
    v = np.array(perseed(csvf, cond, mi, 3, mode))
    return v, float(v.mean()), float(v.std(ddof=1)), float(v.mean())/norm

PANEL = [["a", "b"], ["c", "d"]]
fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.4), sharex=True, sharey=True)
rng = np.random.default_rng(1)
for ri, (rk, rlab, rcol) in enumerate(ROWS):
    for ci, (ck, clab) in enumerate(COLS):
        ax = axes[ri][ci]
        v, m, sd, frac = cell_data(rk, ck)
        figstyle.panel_letter(ax, PANEL[ri][ci], x=-0.06 if ci else -0.30, y=1.01)
        ax.axhline(100, color="0.55", lw=0.8, ls=(0,(4,3)), zorder=1)   # full retention
        ax.axhline(0, color="0.7", lw=0.7, zorder=1)
        norm = NORMAL[rk]
        if v is not None:
            fr = v / norm * 100.0
            jit = (rng.random(len(fr)) - 0.5) * 0.5
            ax.scatter(jit, fr, s=22, color=rcol, alpha=0.8, edgecolor="white",
                       linewidth=0.5, zorder=4)
            se = sd/np.sqrt(len(v))/norm*100.0
            ax.vlines(0, frac*100 - se, frac*100 + se, color=rcol, lw=1.2, zorder=5)
            tag = ""
        else:
            ax.vlines(0, frac*100 - COORD_FROZEN["sd"]/norm*100, frac*100 + COORD_FROZEN["sd"]/norm*100,
                      color=rcol, lw=1.2, zorder=5)
            tag = "  (n = 5 summary)"
        ax.hlines(frac*100, -0.42, 0.42, color=rcol, lw=2.4, zorder=6)
        ax.annotate(f"{frac*100:.0f}%", (0.52, frac*100 + 5), ha="left", va="bottom",
                    fontsize=9.5, fontweight="bold", color=rcol)
        ax.annotate(f"gap {m:+.2f}{tag}", (0.52, frac*100 + 3), ha="left", va="top",
                    fontsize=6.4, color="0.35")
        ax.set_xlim(-0.75, 1.35); ax.set_ylim(-8, 128)
        ax.set_xticks([]); ax.tick_params(axis="x", length=0)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        if ci == 0:
            ax.set_ylabel(rlab, fontweight="bold", color=rcol, labelpad=6)
            ax.set_yticks([0,50,100]); ax.set_yticklabels(["0","50","100"])
        if ri == 0:
            ax.set_title(clab)
# one shared y-axis meaning, placed on the far left outside the row labels
fig.text(0.005, 0.5, "Adherence retained (% of pre-removal)", rotation=90,
         va="center", ha="left", fontsize=8.5)
fig.tight_layout(rect=[0.05, 0, 1, 1])
figstyle.save(fig, "fig5_storage", title="Persistence mechanism")
for rk,_,_ in [(r[0],0,0) for r in ROWS]:
    for ck,_ in COLS:
        v,m,sd,frac = cell_data(rk,ck)
        print(f"  {rk:7} x {ck:8}: gap {m:+.3f}  = {frac*100:4.0f}% of normal"
              + ("  (summary)" if v is None else f"  n={len(v)}"))
