"""Fig 2 -- The dissociation. Three rules at endpoint after oversight lifts (ghost-end gap),
per-seed points, three configurations, non-overlap rendered as a shaded separation band.
Vestige collapses to ~0 (rule gone); the two grounded rules persist, and the top vestige
seed sits below the bottom grounded seed in every configuration. New plot.
  python make_fig2_dissociation.py
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

# canonical cell -> CSV (the dissociation table)
CONFIGS = [
 ("N = 12 agents", [("Vestige","extinct3_cleanghost.csv","1",1,3,"avoid"),
             ("Environmental","extinct3_cleanghost.csv","0",0,3,"avoid"),
             ("Coordination","coord_cleanghost_10.csv","1",1,3,"converge")]),
 ("N = 24 agents", [("Vestige","gen_N24_ghost.csv","1",1,3,"avoid"),
             ("Environmental","gen_N24_poison.csv","0",0,3,"avoid"),
             ("Coordination","gen_N24_coord_cleanghost.csv","1",1,3,"converge")]),
 ("4 berry types", [("Vestige","gen_4b_silly.csv","1",1,4,"avoid"),
              ("Environmental","gen_4b_poison.csv","0",0,4,"avoid"),
              ("Coordination","gen_4b_coord_cleanghost.csv","1",1,4,"converge")]),
]
NORMS = ["Vestige", "Environmental", "Coordination"]
P = figstyle.PALETTE
COL = {"Vestige":P["vestige"], "Environmental":P["environmental"], "Coordination":P["coordination"]}
LAB = {"Vestige":"Vestige\n(silly)", "Environmental":"Environ.\n(poison)", "Coordination":"Coord."}
PANEL = ["a", "b", "c"]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
rng = np.random.default_rng(0)
for pi, (ax, (cfg, cells)) in enumerate(zip(axes, CONFIGS)):
    figstyle.panel_letter(ax, PANEL[pi], x=-0.02 if pi else -0.30, y=1.02)
    vals = {}
    for xpos, (norm, path, cond, mi, nbt, mode) in enumerate(cells):
        v = np.array(perseed(path, cond, mi, nbt, mode)) if os.path.exists(path) else np.array([])
        vals[norm] = v
        jit = (rng.random(len(v)) - 0.5) * 0.20
        ax.scatter(xpos + jit, v, s=17, color=COL[norm], alpha=0.80,
                   edgecolor="white", linewidth=0.4, zorder=4)
        if len(v):
            m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            ax.hlines(m, xpos - 0.28, xpos + 0.28, color=COL[norm], lw=2.2, zorder=5)
            ax.vlines(xpos, m - se, m + se, color=COL[norm], lw=1.1, zorder=5)
            # value printed just past the right end of the mean bar (no spine collision)
            ax.annotate(f"{m:+.2f}", (xpos + 0.33, m), ha="left", va="center",
                        fontsize=7, color=COL[norm], fontweight="bold")
    # non-overlap separation zone: the empty band between the TOP vestige seed and the
    # BOTTOM grounded seed. Every grounded seed clears every vestige seed => distributions
    # separate, not just means. This is Fig 2's job (Fig 3's single-config tiles can't show it).
    if len(vals["Vestige"]) and len(vals["Environmental"]) and len(vals["Coordination"]):
        vmax = float(np.max(vals["Vestige"]))
        gmin = min(float(np.min(vals["Environmental"])), float(np.min(vals["Coordination"])))
        if gmin > vmax:
            # contained band (inset from the panel edges so it doesn't bleed to the frame)
            ax.axhspan(vmax, gmin, xmin=0.04, xmax=0.96, color="#e4efe7", zorder=0)
            ax.hlines([vmax, gmin], -0.40, 2.75, color="0.78", lw=0.5, zorder=0.5)
            # label repeated faintly in every panel so the band is self-explaining anywhere
            ax.annotate("no overlap", (2.62, (vmax + gmin) / 2), ha="right", va="center",
                        fontsize=6.3, color="#40694f", style="italic", zorder=6)
    ax.axhline(0, color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_title(cfg)
    ax.set_xticks(range(3)); ax.set_xticklabels([LAB[n] for n in NORMS], fontsize=7)
    ax.set_xlim(-0.55, 2.9)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Adherence retained after\noversight lifts (per-encounter gap)")
axes[0].set_ylim(-0.05, 0.40)
axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
fig.tight_layout(w_pad=1.0)
figstyle.save(fig, "fig2_dissociation", title="The grounding dissociation")
for cfg, cells in CONFIGS:
    row = []
    for norm, path, cond, mi, nbt, mode in cells:
        v = perseed(path, cond, mi, nbt, mode) if os.path.exists(path) else []
        row.append(f"{norm[:4]} {st.mean(v):+.3f} (n{len(v)})")
    print(f"  {cfg:14}: " + "  ".join(row))
