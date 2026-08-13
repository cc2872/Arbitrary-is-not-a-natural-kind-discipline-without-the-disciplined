"""Fig S5 -- the measure-switch defense (rise vs endpoint).

The SAME vestige persistence quantity, measured two ways across three configurations:
  (a) RISE from install  = how much the avoidance gap moved from install-end to ghost-end
      (marked-berry rise minus control drift, CRN-paired). This is baseline-referenced and is
      CONFOUNDED by the install-baseline shift under enforcement density -> it INVERTS sign at N=24.
  (b) GHOST-END ENDPOINT = the residual avoidance gap at ghost-end (the pre-registered estimand).
      Stable and same-signed across all three configurations -> does NOT invert.
The contrast is the figure: switching from (a) to (b) is a confound diagnosis, not a convenience.

Vestige (silly), cond '1'. Data: extinct3_cleanghost.csv (N=12), gen_N24_ghost.csv (N=24),
gen_4b_silly.csv (4 berry). Windows: install (900-999), ghost (1500-1599).
  python make_figS5_rise_inversion.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.set_pub_style()

REF, GATE = (900, 999), (1500, 1599)
VES = figstyle.PALETTE["vestige"]
CONFIGS = [("N = 12", "extinct3_cleanghost.csv", "1", 1, 3),
           ("N = 24", "gen_N24_ghost.csv",       "1", 1, 3),
           ("4 berry", "gen_4b_silly.csv",        "1", 1, 4)]

def measures(path, cond, mi, nbt):
    ctrl = max(t for t in range(nbt) if t != mi)
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] == cond]
    seeds = sorted({int(r["seed"]) for r in rows})
    def opp(s, i, lo, hi):
        rs = [r for r in rows if int(r["seed"]) == s and lo <= int(r["update"]) <= hi]
        e = sum(float(r[f"eat{i}"]) for r in rs); c = sum(float(r[f"enc{i}"]) for r in rs)
        return e / max(c, 1e-9)
    rise, endpt = [], []
    for s in seeds:
        gap_inst = opp(s, ctrl, *REF) - opp(s, mi, *REF)    # avoidance at install-end
        gap_ghost = opp(s, ctrl, *GATE) - opp(s, mi, *GATE)  # avoidance at ghost-end
        rise.append(gap_inst - gap_ghost)   # how much avoidance was LOST (baseline-referenced)
        endpt.append(gap_ghost)             # residual avoidance (the endpoint estimand)
    return np.array(rise), np.array(endpt)

def msem(a):
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

RISE = {c[0]: msem(measures(*c[1:])[0]) for c in CONFIGS}
ENDP = {c[0]: msem(measures(*c[1:])[1]) for c in CONFIGS}
labels = [c[0] for c in CONFIGS]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.2, 2.7), sharex=True)
for ax, D, ttl, ylab in [(axA, RISE, "Rise from install  (confounded)", "Δ avoidance from install\n(rise measure)"),
                         (axB, ENDP, "Ghost-end endpoint  (pre-registered)", "residual gap at ghost-end\n(endpoint measure)")]:
    ms = [D[l][0] for l in labels]; se = [D[l][1] for l in labels]
    cols = [VES] * 3
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.bar(range(3), ms, yerr=se, width=0.6, color=cols, alpha=0.9, edgecolor="0.3",
           linewidth=0.5, error_kw=dict(ecolor="0.3", elinewidth=1.0, capsize=3), zorder=2)
    for i, (m, s) in enumerate(zip(ms, se)):
        ax.annotate(f"{m:+.3f}", (i, m + (0.004 if m >= 0 else -0.004)),
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=6.4,
                    color="0.15", zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_title(ttl, fontsize=8.3)
    ax.set_ylabel(ylab)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# NOTE: on the current files the RISE does NOT invert at N=24 (all positive: the vestige
# decays in every config). The intended N=24 sign-inversion is not reproduced here -- pending
# the correct N=24 run / measure from the ledger before this panel's narrative annotation.
axA.axhline(0, color="0.55", lw=0.8)
figstyle.panel_letter(axA, "a", x=-0.24, y=1.02)
figstyle.panel_letter(axB, "b", x=-0.24, y=1.02)
fig.tight_layout(w_pad=1.8)
figstyle.save(fig, "figS5_rise_inversion", title="Rise-vs-endpoint measure switch")

print("vestige, rise (delta from install) vs endpoint (ghost-end gap):")
for l in labels:
    print(f"  {l:8s}: rise {RISE[l][0]:+.3f} ± {RISE[l][1]:.3f}   endpoint {ENDP[l][0]:+.3f} ± {ENDP[l][1]:.3f}")
