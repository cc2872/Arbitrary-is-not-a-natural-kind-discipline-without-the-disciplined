"""Fig 4 -- What enforcement is made of: the norm lives on the enforcer's side.

Net taboo decay after each removal gate, per norm (CRN-paired vs the unmarked berry-2
drift, +/-SEM over seeds). The canonical DV is the marked-berry per-encounter RISE across
the switch minus the berry-2 drift (a decay-rate, not an endpoint):
  violator cost gated              -> both norms hold (~0): the gate leaves enforcement live.
  enforcer incentive gated (alone) -> the vestige norm decays while the environmental
  both gated (full removal)           (physical) one relaxes far less -> decay-rate dissociation.
=> the norm is localized in the ENFORCER's incentive, not the violator's cost.

Anchor vocabulary + palette shared with Figs 2/3/5: vestige (silly) grey, environmental
(poison) red. (Supersedes the old pnas_style poison=orange / silly=blue convention, whose
blue collided with coordination.)
Data: extinct3.csv (violator-cost gated), extinct3_enforceronly.csv (enforcer gated),
      extinct3_cleanghost.csv / extinct3_cleanghost_poison10.csv (both gated / full removal).
  python make_fig4_enforcement.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.set_pub_style()

REF, GATE = (900, 999), (1500, 1599)     # last-100 windows: install-end, gate-end
P = figstyle.PALETTE
ENV, VES = P["environmental"], P["vestige"]   # physical/poison (red), social/silly (grey)

def net_decay(path, cond, mi, ci=2):
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] == cond]
    seeds = sorted({int(r["seed"]) for r in rows})
    def opp(s, i, lo, hi):
        rs = [r for r in rows if int(r["seed"]) == s and lo <= int(r["update"]) <= hi]
        e = sum(float(r[f"eat{i}"]) for r in rs); c = sum(float(r[f"enc{i}"]) for r in rs)
        return e / max(c, 1e-9)
    # marked-berry rise across the switch MINUS unmarked berry-2 drift, paired per seed
    v = np.array([(opp(s, mi, *GATE) - opp(s, ci, *GATE)) - (opp(s, mi, *REF) - opp(s, ci, *REF))
                  for s in seeds])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)

GROUPS = ["violator cost\ngated", "enforcer incentive\ngated", "both\ngated"]
NOTE   = ["both hold", "dissociation", "dissociation"]
# (x-group, norm-key, file, condition, marked-idx)
PTS = [
    (0, "ves", "extinct3.csv",                     "1", 1),
    (0, "env", "extinct3.csv",                     "0", 0),
    (1, "ves", "extinct3_enforceronly.csv",        "1", 1),
    (1, "env", "extinct3_enforceronly.csv",        "0", 0),
    (2, "ves", "extinct3_cleanghost.csv",          "1", 1),
    (2, "env", "extinct3_cleanghost_poison10.csv", "0", 0),
]
STYLE = {"ves": dict(color=VES, label="vestige (silly)",        dx=-0.15),
         "env": dict(color=ENV, label="environmental (poison)", dx=+0.15)}

# compute once, keyed by norm
data = {"ves": [], "env": []}
for g, key, path, cond, mi in PTS:
    m, se, n = net_decay(path, cond, mi)
    data[key].append((g + STYLE[key]["dx"], m, se, n))

fig, ax = plt.subplots(figsize=(4.0, 2.9))
ax.axhline(0, color="0.6", lw=0.8, zorder=1)
ax.annotate("holds", (-0.44, 0.002), fontsize=6, color="0.55", va="bottom", ha="left")
# faint connectors: same norm across the three gates (shows the rise appears only for
# the vestige, and only once the enforcer incentive is gated)
for key in ("ves", "env"):
    xs = [d[0] for d in data[key]]; ys = [d[1] for d in data[key]]
    ax.plot(xs, ys, color=STYLE[key]["color"], lw=0.8, alpha=0.35, zorder=2)
for key in ("ves", "env"):
    st = STYLE[key]
    for (x, m, se, n) in data[key]:
        ax.errorbar(x, m, yerr=se, fmt="o", color=st["color"], ms=5, lw=1.1, capsize=2.5,
                    mec="0.3", mew=0.4, zorder=3)
    ax.plot([], [], "o", color=st["color"], mec="0.3", mew=0.4, label=st["label"])
    for (x, m, se, n) in data[key]:                        # value labels, offset outward
        ax.annotate(f"{m:+.3f}", (x + (0.07 if key == "env" else -0.07), m),
                    fontsize=6.2, color=st["color"], ha="left" if key == "env" else "right",
                    va="center", zorder=4)

for g, note in zip([0, 1, 2], NOTE):
    ax.text(g, 0.076, note, ha="center", fontsize=6.6, color="0.4", style="italic")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(GROUPS)
ax.set_xlim(-0.55, 2.6); ax.set_ylim(-0.028, 0.083)
ax.set_ylabel("Net taboo decay after removal\n(Δ per-encounter gap, CRN-paired)")
ax.tick_params(axis="x", length=0)
ax.legend(loc="center left", handlelength=1.0, labelspacing=0.35, borderaxespad=0.4)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
figstyle.save(fig, "fig4_enforcement", title="What enforcement is made of")

print("net taboo decay (mean, sem, n, margin):")
for key in ("ves", "env"):
    for (x, m, se, n), grp in zip(data[key], ["violator", "enforcer", "both"]):
        print(f"  {grp:9s} {key}: {m:+.3f} +/- {se:.3f}  (n={n}, margin {m/se:+.1f})")
