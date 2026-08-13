"""Table 1 -- the anchor space (conceptual companion to Fig 3).
Four enforced-norm types, each DEFINED by which grounding knockout kills it. Three rows
established (environmental, coordination, pure vestige); shibboleth predicted (stretch).
Rendered as a styled table figure; palette shared with the main figures.
  python make_table1_anchors.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle
figstyle.set_pub_style()
P = figstyle.PALETTE

KILL_BG = "#f4ded9"    # 'dies here' cell: pink/coral tint background (all text stays black)
DASH = "0.6"

COLS = ["Anchor", "Grounded in", "Remove enforcement\n(ghost)",
        "Remove its own\ngrounding", "Collapse shape"]
XL = [0.005, 0.210, 0.410, 0.600, 0.800]   # column left edges (axes fraction)
XW = [0.200, 0.195, 0.185, 0.195, 0.195]   # column widths
def cx(j): return XL[j] + XW[j] / 2

# cells = [grounded_in, ghost, own-grounding, shape]; kill = index into cells that "dies"
ROWS = [
    dict(anchor="environmental", sub="(poison)", color=P["environmental"], predicted=False,
         cells=["external payoff cost\non the act", "persists", "decays\n(hazard off)",
                "monotone, no tip"], kill=2),
    dict(anchor="coordination", sub="", color=P["coordination"], predicted=False,
         cells=["payoff from\nmatching others", "persists\n(self-sustaining)",
                "decays\n(returns flattened)", "sharp, hysteretic\n(critical mass)"], kill=2),
    dict(anchor="pure vestige", sub="(silly)", color="#6b7075", predicted=False,
         cells=["nothing but\nenforcement", "decays", "(nothing\nto remove)",
                "monotone,\nno critical point"], kill=1),
    dict(anchor="shibboleth", sub="(predicted)", color="#8a7aa0", predicted=True,
         cells=["in-group\nsignalling benefit", "persists if\ninteraction live",
                "decays\n(group removed)", "bistable by group"], kill=2),
]

fig, ax = plt.subplots(figsize=(7.4, 2.75))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

HEAD_Y = 0.88
ROW_Y = [0.68, 0.50, 0.32, 0.12]
RH = 0.165

# header band
ax.add_patch(Rectangle((0.0, HEAD_Y - 0.075), 1.0, 0.15, facecolor="0.93",
                       edgecolor="none", zorder=0))
for j, c in enumerate(COLS):
    ax.text(cx(j) if j else XL[0] + 0.01, HEAD_Y, c, fontsize=7.6, fontweight="bold",
            ha="center" if j else "left", va="center", color="black")
ax.plot([0, 1], [HEAD_Y - 0.075, HEAD_Y - 0.075], color="0.5", lw=0.8)

for ri, row in enumerate(ROWS):
    y = ROW_Y[ri]
    it = "italic" if row["predicted"] else "normal"
    # dashed rule above the predicted (stretch) row to fence established vs predicted
    if row["predicted"]:
        ax.plot([0, 1], [y + RH / 2 + 0.005, y + RH / 2 + 0.005], color=DASH, lw=0.7,
                ls=(0, (4, 3)))
    # kill cell highlight
    kj = row["kill"] + 1
    ax.add_patch(Rectangle((XL[kj], y - RH / 2), XW[kj], RH, facecolor=KILL_BG,
                           edgecolor="none", zorder=0, alpha=0.9 if not row["predicted"] else 0.6))
    # anchor name (black, bold) + sub-label
    ax.text(XL[0] + 0.01, y + (0.028 if row["sub"] else 0), row["anchor"], fontsize=8,
            fontweight="bold", style=it, color="black", ha="left", va="center")
    if row["sub"]:
        ax.text(XL[0] + 0.01, y - 0.035, row["sub"], fontsize=6.3, style="italic",
                color="black", ha="left", va="center")
    # the three knockout/shape cells (all black; the kill cell keeps its pink background + bold)
    for c, txt in enumerate(row["cells"]):
        j = c + 1
        is_kill = (c == row["kill"])
        fw = "bold" if is_kill else "normal"
        ax.text(cx(j), y, txt, fontsize=6.7, ha="center", va="center", color="black",
                fontweight=fw, style=it, linespacing=1.15)
    if ri < len(ROWS) - 1 and not ROWS[ri + 1]["predicted"]:
        ax.plot([0, 1], [(ROW_Y[ri] + ROW_Y[ri + 1]) / 2, (ROW_Y[ri] + ROW_Y[ri + 1]) / 2],
                color="0.88", lw=0.6)

ax.text(0.005, 0.005, "Coral marks the knockout that kills each type. Only the pure vestige "
        "dies to enforcement removal; the grounded types survive it and each dies to its own "
        "distinct grounding. Shibboleth is a predicted stretch row.",
        fontsize=6, color="black", ha="left", va="bottom", style="italic")

fig.tight_layout(pad=0.4)
figstyle.save(fig, "table1_anchors", title="The anchor space")
print("Table 1 rendered: 3 established rows + shibboleth (predicted).")
