"""Fig S7 -- the gating audit (validity control).

Zaps per episode across the enforcement switch (update 1000), one line per condition. The
gated regimes (full removal, enforcer-incentive removal) flatline to ~zero at the switch;
the deliberately-retained regimes (violator-cost-only, timeout-only) stay live by design.
=> enforcement genuinely ceased where gated, so every decay reading reflects loss of
oversight, not residual enforcement.

Data: extinct3_{cleanghost,enforceronly,timeoutonly}.csv + extinct3.csv (violator-only),
'zaps' column, condition '1'. Switch at update 1000.
  python make_figS7_gating_audit.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.set_pub_style()

SWITCH = 1000
# (file, label, colour, gated?, label_y override)  -- neutral categorical palette
SERIES = [
    ("extinct3_cleanghost.csv",   "full removal",        "#222222", True,  -12),
    ("extinct3_enforceronly.csv", "enforcer-only",       "#3a86a8", True,  28),
    ("extinct3.csv",              "violator-only",       "#c1554a", False, None),
    ("extinct3_timeoutonly.csv",  "timeout-only",        "#d19a3c", False, None),
]

def zaps_traj(path, cond="1"):
    rows = [r for r in csv.DictReader(open(path)) if r["condition"] == cond]
    by_u = {}
    for r in rows:
        by_u.setdefault(int(r["update"]), []).append(float(r["zaps"]))
    us = np.array(sorted(by_u))
    z = np.array([np.mean(by_u[u]) for u in us])
    if len(z) >= 11:                      # light smoothing for legibility
        k = np.ones(11) / 11
        z = np.convolve(np.pad(z, 5, mode="edge"), k, mode="valid")
    return us, z

fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.set_xlim(0, 1600 + 340)              # room for right-edge labels
ax.set_ylim(-18, 715)
ax.axvline(SWITCH, color="0.6", lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(SWITCH + 14, 705, "switch", fontsize=6.5, color="0.5", va="top", ha="left")
for path, lab, col, gated, laby in SERIES:
    us, z = zaps_traj(path)
    ax.plot(us, z, color=col, lw=1.3, zorder=3, solid_capstyle="round")
    ly = z[-1] if laby is None else laby
    ax.annotate(lab, (us[-1] + 12, ly), fontsize=6.6, color=col, va="center", ha="left",
                fontweight="bold" if not gated else "normal")
ax.set_xlabel("update  (install 0–999  |  ghost 1000–1599)")
ax.set_ylabel("enforcement  (zaps per episode)")
ax.axhline(0, color="0.8", lw=0.7, zorder=0)
# annotate the two behaviours
ax.text(1300, 545, "retained by design", fontsize=6.6, color="0.35", style="italic", ha="center")
ax.text(1300, 55, "gated → enforcement stops", fontsize=6.6, color="0.35", style="italic", ha="center")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
figstyle.save(fig, "figS7_gating_audit", title="Gating audit")

print("mean zaps  install(900-999) -> ghost(1500-1599), cond '1':")
for path, lab, *_ in SERIES:
    us, z = zaps_traj(path)
    inst = z[(us >= 900) & (us <= 999)].mean(); gh = z[(us >= 1500) & (us <= 1599)].mean()
    print(f"  {lab:16s}: {inst:6.1f} -> {gh:6.1f}")
