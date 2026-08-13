"""Fig S3 -- the affordability / emergent-install result.

(a) Punishing harder does NOT install the rule: sweeping the zap penalty c_zapped {2,15,35}
    weakens the installed rule (poison diff-in-diff +0.162 -> +0.026 -> +0.004) AND collapses
    collective return (+8.4 -> -26.8 -> -49.5) together -- harder punishment wrecks the economy.
(b) The arbitrary (silly) rule installs ONLY when compliance is affordable: null in the
    two-berry world (obeying = starving on poison), +0.098 once a third, harmless berry exists.
This was discovered, not designed -- an emergent property of the substrate.

Data: zap_c{2,15,35}.csv (2-berry, cond 0/none + berry-1 control); opp2_8.75.csv (2-berry silly);
multiberry.csv (3-berry silly). All last-100 (1400-1499), CRN diff-in-diff vs the neutral berry.
  python make_figS3_affordability.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.set_pub_style()

P = figstyle.PALETTE
POISON, VES = P["environmental"], P["vestige"]
RET_COL = "#6b7075"   # collective return in a neutral tone ('the cost')

def rows_of(path): return list(csv.DictReader(open(path)))
def opp(rows, cond, i, lo=1400, hi=1499):
    rs = [r for r in rows if r["condition"] == cond]; seeds = sorted({int(r["seed"]) for r in rs})
    out = []
    for s in seeds:
        d = [r for r in rs if int(r["seed"]) == s and lo <= int(r["update"]) <= hi]
        e = sum(float(r[f"eat{i}"]) for r in d); c = sum(float(r[f"enc{i}"]) for r in d)
        out.append(e / max(c, 1e-9))
    return np.array(out)
def ret(rows, cond, lo=1400, hi=1499):
    rs = [float(r["ret"]) for r in rows if r["condition"] == cond and lo <= int(r["update"]) <= hi]
    return np.mean(rs)
def did(rows, mi, ctrl, marked_cond, base="none"):   # diff-in-diff install strength
    return (opp(rows, base, mi).mean() - opp(rows, marked_cond, mi).mean()) \
         - (opp(rows, base, ctrl).mean() - opp(rows, marked_cond, ctrl).mean())

# -- (a) czapped sweep --
CZ = [(2, "zap_c2.csv"), (15, "zap_c15.csv"), (35, "zap_c35.csv")]
cvals = [c for c, _ in CZ]
strength = [did(rows_of(f), 0, 1, "0") for c, f in CZ]   # poison-specific (ctrl = safe berry 1)
returns = [ret(rows_of(f), "0") for c, f in CZ]

# -- (b) affordability: 2-berry silly (opp2) vs 3-berry silly (multiberry) --
silly_2b = did(rows_of("opp2_8.75.csv"), 1, 0, "1")      # 2-berry: control is berry-0
silly_3b = did(rows_of("multiberry.csv"), 1, 2, "1")     # 3-berry: control is neutral berry-2

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))

# Panel A: dual axis
x = np.arange(3)
axA.plot(x, strength, "-o", color=POISON, ms=5, lw=1.4, mec="0.3", mew=0.4, zorder=3)
for xi, s in zip(x, strength):
    axA.annotate(f"{s:+.3f}", (xi, s), textcoords="offset points", xytext=(-7, 7),
                 ha="right", va="bottom", fontsize=6.2, color=POISON)
axA.set_ylabel("poison rule strength\n(diff-in-diff)", color=POISON)
axA.tick_params(axis="y", labelcolor=POISON)
axA.set_ylim(-0.02, 0.20)
axA.axhline(0, color="0.75", lw=0.7)
axR = axA.twinx()
axR.plot(x, returns, "--s", color=RET_COL, ms=4.5, lw=1.3, mec="0.3", mew=0.4, zorder=3)
for xi, r in zip(x, returns):
    axR.annotate(f"{r:+.1f}", (xi, r), textcoords="offset points", xytext=(7, -7),
                 ha="left", va="top", fontsize=6.2, color=RET_COL)
axR.set_ylabel("collective return", color=RET_COL, rotation=270, labelpad=12)
axR.tick_params(axis="y", labelcolor=RET_COL)
axR.axhline(0, color=RET_COL, lw=0.6, ls=(0, (2, 2)), alpha=0.5)
axR.set_ylim(-62, 20)
axR.spines["top"].set_visible(False)
axA.set_xticks(x); axA.set_xticklabels([f"{c}" for c in cvals])
axA.set_xlim(-0.42, 2.42)               # left padding so the up-left value labels clear the axis
axA.set_xlabel("zap penalty  $c_{\\mathrm{zapped}}$")
axA.set_title("Punishing harder wrecks the economy", fontsize=8.2)
axA.tick_params(axis="x", length=0)
axA.spines["top"].set_visible(False)
figstyle.panel_letter(axA, "a", x=-0.22, y=1.03)

# Panel B: affordability bars
vals = [silly_2b, silly_3b]
axB.axhline(0, color="0.6", lw=0.8)
bars = axB.bar([0, 1], vals, width=0.6, color=VES, alpha=0.9, edgecolor="0.3", linewidth=0.5)
for xi, v in zip([0, 1], vals):
    axB.annotate(f"{v:+.3f}", (xi, v + (0.004 if v >= 0 else -0.004)), ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=6.6, color="0.15", fontweight="bold")
axB.set_xticks([0, 1])
axB.set_xticklabels(["2 berries\n(obey = starve)", "3 berries\n(obey affordable)"])
axB.set_ylabel("silly rule strength\n(diff-in-diff)")
axB.set_ylim(-0.03, 0.12)
axB.set_title("Installs only when affordable", fontsize=8.2)
axB.tick_params(axis="x", length=0)
for sp in ("top", "right"): axB.spines[sp].set_visible(False)
figstyle.panel_letter(axB, "b", x=-0.28, y=1.03)

fig.tight_layout(w_pad=2.2)
figstyle.save(fig, "figS3_affordability", title="Affordability / emergent install")

print("czapped:", [f"c={c}: str {s:+.3f} ret {r:+.1f}" for c, s, r in zip(cvals, strength, returns)])
print(f"affordability: 2-berry silly {silly_2b:+.3f}  |  3-berry silly {silly_3b:+.3f}")
