"""Fig 8 -- the matching (coordination) rule has an operating window, not a knife edge.
Three quantities vs the matching benefit coord_k, all monotone in the single parameter:
  * convergence PREFERENCE   (baseline, no enforcement; opp1-opp2 at install-end)  -- installs, grows with k
  * flatten-knockout COLLAPSE (flat run rel_decay, converge-mode; >0 = convergence lost) -- always collapses
  * persistence under OVERSIGHT REMOVAL (cleanghost rel_decay; <=0 = persists)     -- persists only at high k
Data from coord_baseline/cleanghost/flat_k{0p5,1p0,2p0}.csv (n=5) via the paper2_summary metric.
Marks the value used for the coordination configuration (k=2.0) and the persistence window edge.
"""
import csv, statistics as st, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

def load(p): return list(csv.DictReader(open(p)))
def seeds(r, c): return sorted(set(int(x['seed']) for x in r if x['condition'] == c))
def opp(r, c, s, i, lo, hi):
    rs = [x for x in r if x['condition'] == c and int(x['seed']) == s and lo <= int(x['update']) < hi]
    e = sum(float(x['eat%d' % i]) for x in rs); d = sum(float(x['enc%d' % i]) for x in rs)
    return e / max(d, 1e-9)
def ms(v): return st.mean(v), (st.stdev(v) / len(v) ** .5 if len(v) > 1 else 0.0)
def pref(kf):                              # baseline: opp1-opp2 at install-end (grounding alone)
    r = load('coord_baseline_k%s.csv' % kf); S = seeds(r, 'none'); um = max(int(x['update']) for x in r)
    return ms([opp(r, 'none', s, 1, um - 99, um + 1) - opp(r, 'none', s, 2, um - 99, um + 1) for s in S])
def reldecay(path):                         # converge-mode rel_decay vs companion berry 2 (>0 = convergence lost)
    r = load(path); c = '1'; S = seeds(r, c)
    v = [-((opp(r, c, s, 1, 1500, 1600) - opp(r, c, s, 1, 900, 1000))
           - (opp(r, c, s, 2, 1500, 1600) - opp(r, c, s, 2, 900, 1000))) for s in S]
    return ms(v)

KS = [0.5, 1.0, 2.0]; KF = ['0p5', '1p0', '2p0']
prefm = [pref(k) for k in KF]
cgm = [reldecay('coord_cleanghost_k%s.csv' % k) for k in KF]
flm = [reldecay('coord_flat_k%s.csv' % k) for k in KF]
pv = np.array([m for m, _ in prefm]); pe = np.array([e for _, e in prefm])
cv = np.array([m for m, _ in cgm]);  ce = np.array([e for _, e in cgm])
fv = np.array([m for m, _ in flm]);  fe = np.array([e for _, e in flm])
x = np.array(KS)
# persistence window edge: linear crossing of the cleanghost curve through 0 (between k=1 and k=2)
i = int(np.where(cv[:-1] * cv[1:] <= 0)[0][0])
xcross = x[i] + (0 - cv[i]) * (x[i + 1] - x[i]) / (cv[i + 1] - cv[i])

F.set_pub_style()
BLUE, RED, GREEN, ZERO = F.PALETTE['coordination'], F.PALETTE['collapse'], F.PALETTE['keep'], F.PALETTE['zero']
fig, ax = plt.subplots(figsize=(3.5, 2.9))

# persistence operating window (cleanghost Delta <= 0 while flat Delta > 0)
ax.axvspan(xcross, 2.18, color=BLUE, alpha=0.07, lw=0, zorder=0)
ax.text(xcross + 0.03, 0.10, 'persistence\nwindow', ha='left', va='center', rotation=90,
        fontsize=6.3, color=BLUE, alpha=.9)
ax.axhline(0, color=ZERO, lw=0.7, ls=(0, (4, 3)), zorder=1)
# persist/lost semantics: place at the 0-line's left end, clear of every curve (all >=0.06 at k=0.5)
ax.text(0.42, 0.028, 'Δ>0  convergence lost', fontsize=6.1, color=ZERO, va='center', ha='left')
ax.text(0.42, -0.028, 'Δ<0  persists',        fontsize=6.1, color=ZERO, va='center', ha='left')

kw = dict(lw=1.4, capsize=2, elinewidth=0.8, ms=5, zorder=3, clip_on=False)
ax.errorbar(x, pv, yerr=pe, marker='o', color=BLUE, label='preference (installs)', **kw)
ax.errorbar(x, fv, yerr=fe, marker='^', color=RED,  label='flatten-off: collapse', **kw)
ax.errorbar(x, cv, yerr=ce, marker='s', color=GREEN, label='oversight-off: persist', **kw)

# mark the value used for the coordination configuration
ax.axvline(2.0, color='0.35', lw=0.7, ls=':', zorder=1)
ax.annotate('value used\n(k = 2.0)', xy=(2.0, -0.030), xytext=(2.0, -0.030),
            ha='center', va='top', fontsize=6.6, color='0.2')
# the dissociation = vertical gap between collapse and persistence at the used k
ax.annotate('', xy=(2.10, fv[-1]), xytext=(2.10, cv[-1]),
            arrowprops=dict(arrowstyle='<->', color='0.45', lw=0.8))
ax.text(2.14, (fv[-1] + cv[-1]) / 2, 'dissociation', rotation=90, va='center', ha='left',
        fontsize=6.4, color='0.35')

ax.set_xlim(0.35, 2.35); ax.set_ylim(-0.055, 0.20)
ax.set_xticks([0.5, 1.0, 1.5, 2.0])
ax.set_xlabel('matching benefit  $coord\\_k$')
ax.set_ylabel('convergence advantage / decay  $\\Delta$  (opp units)')
# legend ABOVE the axes (interior is full -- three curves fan across it), one compact row
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, columnspacing=1.1,
          handlelength=1.3, handletextpad=0.4, borderaxespad=0.0)
fig.tight_layout()
F.save(fig, 'fig8_coord_window', title='Fig 8 -- coordination rule operating window')
print('preference :', [f'{m:+.3f}' for m, _ in prefm])
print('cleanghost :', [f'{m:+.3f}' for m, _ in cgm], ' (persist where <=0)')
print('flat       :', [f'{m:+.3f}' for m, _ in flm])
print('window edge (cleanghost crosses 0): k =', round(float(xcross), 2))
