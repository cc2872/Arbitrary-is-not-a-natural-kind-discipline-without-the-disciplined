"""Fig S10 -- the decode/ablate null (Arm 3, pre-registered kill condition firing).

The norm direction IS linearly decodable from the GRU hidden state (esp. environmental),
but ABLATING the decoded direction moves behavior no more than ablating a RANDOM direction.
=> representation is present but not shown causal: the pre-registered ablation kill condition
fires, and the internalization-as-causal claim is honestly withdrawn (title stays behavioral).

Data: probe_arm3_{poison,coord,vestige}.json (5 seeds each; r2 = linear-probe decode R^2 of
the norm direction; move_decoded / move_random = behavioural gap change when the decoded vs a
random carry direction is ablated). Palette shared with the main figures.
  python make_figS10_decode_ablate.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.set_pub_style()

P = figstyle.PALETTE
# (key, file, label, color) in main-text row order: environmental, coordination, vestige
NORMS = [("env", "probe_arm3_poison.json",  "environmental\n(poison)",  P["environmental"]),
         ("coord", "probe_arm3_coord.json", "coordination",             P["coordination"]),
         ("ves", "probe_arm3_vestige.json", "vestige\n(silly)",         P["vestige"])]

def load(path):
    d = json.load(open(path))
    r2 = np.array([s["r2"] for s in d])
    mdec = np.abs([s["move_decoded"] for s in d])   # |behaviour move| under decoded ablation
    mrnd = np.abs([s["move_random"] for s in d])    # |behaviour move| under random ablation
    return r2, mdec, mrnd

def msem(a):
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.2, 2.7))

# -- Panel A: decode R^2 by norm --------------------------------------------------
for i, (k, f, lab, col) in enumerate(NORMS):
    r2, _, _ = load(f)
    m, se = msem(r2)
    axA.bar(i, m, width=0.62, color=col, alpha=0.85, edgecolor="0.3", linewidth=0.5, zorder=2)
    axA.errorbar(i, m, yerr=se, fmt="none", ecolor="0.25", elinewidth=1.0, capsize=3, zorder=3)
    axA.scatter(np.full(len(r2), i) + (np.linspace(-0.16, 0.16, len(r2))), r2, s=9,
                color="white", edgecolor="0.35", linewidth=0.5, zorder=4)
axA.set_xticks(range(3)); axA.set_xticklabels([n[2] for n in NORMS])
axA.set_ylim(0, 1.0); axA.set_ylabel("Decode $R^2$ of the norm direction")
axA.axhline(0, color="0.6", lw=0.7)
axA.set_title("Decodable", fontsize=8.5)
axA.tick_params(axis="x", length=0)
figstyle.panel_letter(axA, "a", x=-0.20, y=1.02)
for sp in ("top", "right"): axA.spines[sp].set_visible(False)

# -- Panel B: ablation behaviour-move, decoded vs random --------------------------
for i, (k, f, lab, col) in enumerate(NORMS):
    _, mdec, mrnd = load(f)
    md, sd = msem(mdec); mr, sr = msem(mrnd)
    axB.errorbar(i - 0.16, md, yerr=sd, fmt="o", color=col, ms=5, mec="0.3", mew=0.4,
                 capsize=2.5, lw=1.0, zorder=3)
    axB.errorbar(i + 0.16, mr, yerr=sr, fmt="o", color="white", mec="0.45", mew=0.9, ms=5,
                 capsize=2.5, ecolor="0.55", lw=1.0, zorder=3)
axB.set_xticks(range(3)); axB.set_xticklabels([n[2] for n in NORMS])
axB.set_xlim(-0.6, 2.6); axB.set_ylim(0, 0.058)
axB.set_ylabel("|behaviour move| under ablation")
axB.set_title("Not causal: ablation ≈ random", fontsize=8.5)
axB.tick_params(axis="x", length=0)
# legend: decoded (filled, norm colour) vs random (hollow)
axB.plot([], [], "o", color="0.4", mec="0.3", mew=0.4, label="decoded direction")
axB.plot([], [], "o", color="white", mec="0.45", mew=0.9, label="random direction")
axB.legend(loc="upper left", handlelength=1.0, labelspacing=0.3, borderaxespad=0.3)
figstyle.panel_letter(axB, "b", x=-0.20, y=1.02)
for sp in ("top", "right"): axB.spines[sp].set_visible(False)

fig.tight_layout(w_pad=1.6)
figstyle.save(fig, "figS10_decode_ablate", title="Decode/ablate null (Arm 3)")

print("Arm-3 decode/ablate (mean over 5 seeds):")
for k, f, lab, col in NORMS:
    r2, mdec, mrnd = load(f)
    print(f"  {k:6s}: R2={r2.mean():.2f}  |move_decoded|={mdec.mean():.4f}  "
          f"|move_random|={mrnd.mean():.4f}  (decoded/random={mdec.mean()/mrnd.mean():.2f})")
