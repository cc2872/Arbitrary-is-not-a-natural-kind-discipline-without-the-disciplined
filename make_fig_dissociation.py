"""fig_dissociation.pdf -- the flagship 3x3 persistence dissociation with per-seed points.
Ghost-end gap (norm present = positive), oriented per norm direction (avoidance =
control-marked; convergence = marked-control). Vestige collapses to ~0 (norm gone); the
two grounded norms remain. One panel per configuration; per-seed points + mean.
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def perseed(path, cond, mi, nbt, mode, lo=1500, hi=1600):
    r=[x for x in csv.DictReader(open(path)) if x['condition']==cond]
    S=sorted(set(int(x['seed']) for x in r)); ctrl=max(t for t in range(nbt) if t!=mi)
    def o(s,i):
        rs=[x for x in r if int(x['seed'])==s and lo<=int(x['update'])<hi]
        e=sum(float(x[f'eat{i}']) for x in rs); c=sum(float(x[f'enc{i}']) for x in rs); return e/max(c,1e-9)
    return [ (o(s,ctrl)-o(s,mi)) if mode=='avoid' else (o(s,mi)-o(s,ctrl)) for s in S]

# canonical cell -> CSV (per the dissociation table / verification pass)
CONFIGS = [
 ("N = 12", [("Vestige","extinct3_cleanghost.csv","1",1,3,"avoid"),
             ("Environmental","extinct3_cleanghost.csv","0",0,3,"avoid"),
             ("Coordination","coord_cleanghost_10.csv","1",1,3,"converge")]),
 ("N = 24", [("Vestige","gen_N24_ghost.csv","1",1,3,"avoid"),
             ("Environmental","gen_N24_poison.csv","0",0,3,"avoid"),
             ("Coordination","gen_N24_coord_cleanghost.csv","1",1,3,"converge")]),
 ("4-berry", [("Vestige","gen_4b_silly.csv","1",1,4,"avoid"),
              ("Environmental","gen_4b_poison.csv","0",0,4,"avoid"),
              ("Coordination","gen_4b_coord_cleanghost.csv","1",1,4,"converge")]),
]
NORMS = ["Vestige", "Environmental", "Coordination"]
COL = {"Vestige":"#9aa0a6", "Environmental":"#d1495b", "Coordination":"#2e6f95"}  # vestige muted; grounded saturated

plt.rcParams.update({"font.size":8, "axes.linewidth":0.7, "xtick.major.width":0.7,
                     "ytick.major.width":0.7, "pdf.fonttype":42, "ps.fonttype":42})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
rng = np.random.default_rng(0)
for ax, (cfg, cells) in zip(axes, CONFIGS):
    for xpos, (norm, path, cond, mi, nbt, mode) in enumerate(cells):
        v = np.array(perseed(path, cond, mi, nbt, mode)) if os.path.exists(path) else np.array([])
        jit = (rng.random(len(v))-0.5)*0.22
        ax.scatter(xpos+jit, v, s=14, color=COL[norm], alpha=0.65, edgecolor="none", zorder=3)
        if len(v):
            m = v.mean()
            ax.hlines(m, xpos-0.28, xpos+0.28, color=COL[norm], lw=2.2, zorder=4)
            ax.annotate(f"{m:+.2f}", (xpos-0.32, m), ha="right", va="center",
                        fontsize=6.8, color=COL[norm])
    ax.axhline(0, color="0.6", lw=0.8, ls=(0,(4,3)), zorder=1)
    ax.set_title(cfg, fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Vest.","Env.","Coord."], fontsize=7.5)
    ax.set_xlim(-0.6, 2.6)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Ghost-end gap\n(norm persistence)")
axes[0].set_ylim(-0.05, 0.38)
# legend OUTSIDE the panels (below), so it never collides with the 4-berry cluster
handles=[plt.Line2D([],[],marker='o',ls='',color=COL[n],label=n,ms=5) for n in NORMS]
fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.08), handletextpad=0.3, columnspacing=1.6)
fig.suptitle("Persistence under oversight removal: vestige collapses, grounded norms remain",
             fontsize=9, y=1.03)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("fig_dissociation.pdf", bbox_inches="tight")
fig.savefig("fig_dissociation.png", dpi=200, bbox_inches="tight")
print("wrote fig_dissociation.pdf + .png")
# also dump the per-cell means for the caption
print("\nper-cell means (ghost-end gap):")
for cfg, cells in CONFIGS:
    row=[]
    for norm,path,cond,mi,nbt,mode in cells:
        v=perseed(path,cond,mi,nbt,mode) if os.path.exists(path) else []
        row.append(f"{norm[:4]} {st.mean(v):+.3f} (n{len(v)})")
    print(f"  {cfg:8}: "+"  ".join(row))
