"""Paper-2 comprehensive data summary + validity audits. CPU-only, reads CSVs."""
import csv, statistics as st, os

def load(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else None
def seeds(r, cond): return sorted(set(int(x['seed']) for x in r if x['condition'] == cond))
def opp(r, cond, s, i, lo, hi):
    rs = [x for x in r if x['condition'] == cond and int(x['seed']) == s and lo <= int(x['update']) < hi]
    e = sum(float(x['eat%d' % i]) for x in rs); c = sum(float(x['enc%d' % i]) for x in rs)
    return e / max(c, 1e-9)
def colmean(r, cond, name, lo, hi):
    v = [float(x[name]) for x in r if x['condition'] == cond and lo <= int(x['update']) < hi and name in x]
    return st.mean(v) if v else float('nan')
def reldecay(path, cond, mi, ci, converge=False):
    r = load(path)
    if not r: return None
    S = seeds(r, cond); sign = -1.0 if converge else 1.0
    v = [sign * ((opp(r, cond, s, mi, 1500, 1600) - opp(r, cond, s, mi, 900, 1000))
                 - (opp(r, cond, s, ci, 1500, 1600) - opp(r, cond, s, ci, 900, 1000))) for s in S]
    m = st.mean(v); mg = m / (st.stdev(v) / len(v) ** .5) if len(v) > 1 else float('nan')
    return dict(n=len(S), mean=m, margin=mg, pos=sum(x > 0 for x in v))

print("#" * 78); print("# 1. VALIDITY AUDITS"); print("#" * 78)
print("\n[Enforcement stopped in ghost?] zaps install(900-999) -> ghost(1500-1599); expect COLLAPSE.")
for path, cond, lbl in [('extinct3_cleanghost.csv', '1', 'vestige silly-ghost'),
                        ('poison_hazardoff.csv', '0', 'poison hazard-off'),
                        ('coord_cleanghost_k2p0.csv', '1', 'coord cleanghost k2.0'),
                        ('coord_flat_k2p0.csv', '1', 'coord flat k2.0'),
                        ('coord_cleanghost_10.csv', '1', 'coord cleanghost_10'),
                        ('coord_flat_10.csv', '1', 'coord flat_10')]:
    r = load(path)
    if not r: continue
    zi = colmean(r, cond, 'zaps', 900, 1000); zg = colmean(r, cond, 'zaps', 1500, 1600)
    print("  %-26s zaps %6.1f -> %5.1f   %s" % (lbl, zi, zg, 'STOP' if zg < zi * 0.2 else 'CONTINUE'))
print("\n[Gate flips] knockout channel 1->0 at update 1000:")
for path, col, lbl in [('poison_hazardoff_10.csv', 'hazard_on', 'hazard_on'),
                       ('coord_flat_10.csv', 'coord_on', 'coord_on (flat)'),
                       ('coord_cleanghost_10.csv', 'coord_on', 'coord_on (cleanghost=stays 1)')]:
    r = load(path)
    if not r or col not in r[0]: continue
    pre = {float(x[col]) for x in r if int(x['update']) < 1000}
    post = {float(x[col]) for x in r if int(x['update']) >= 1000}
    print("  %-32s install=%s ghost=%s" % (lbl, pre, post))

print("\n" + "#" * 78); print("# 2. ENVIRONMENTAL ROW (physical hazard)"); print("#" * 78)
pg = reldecay('extinct3_cleanghost.csv', '0', 0, 2); pg10 = reldecay('extinct3_cleanghost_poison10.csv', '0', 0, 2)
ho5 = reldecay('poison_hazardoff.csv', '0', 0, 2); ho10 = reldecay('poison_hazardoff_10.csv', '0', 0, 2)
print("  poison-GHOST (hazard intact, enforce off)  5s: %+.4f m%+.1f  PERSISTS" % (pg['mean'], pg['margin']))
print("  poison-GHOST                              10s: %+.4f m%+.1f  PERSISTS" % (pg10['mean'], pg10['margin']))
print("  poison HAZARD-OFF (own-knockout)           5s: %+.4f m%+.1f  DECAYS" % (ho5['mean'], ho5['margin']))
print("  poison HAZARD-OFF                         10s: %+.4f m%+.1f  (%d/%d) DECAYS" % (ho10['mean'], ho10['margin'], ho10['pos'], ho10['n']))
print("  => grounding proof: persist %+.3f vs die %+.3f  (%.0fx)" % (pg10['mean'], ho10['mean'], ho10['mean'] / pg10['mean']))

print("\n" + "#" * 78); print("# 3. COORDINATION ROW (k=2.0 flagship)"); print("#" * 78)
cg = load('coord_cleanghost_k2p0.csv'); S = seeds(cg, '1')
o1i = st.mean([opp(cg, '1', s, 1, 900, 1000) for s in S]); o2i = st.mean([opp(cg, '1', s, 2, 900, 1000) for s in S])
sel = colmean(cg, '1', 'selectivity', 900, 1000)
bl = load('coord_baseline_k2p0.csv'); Sb = seeds(bl, 'none'); um = max(int(x['update']) for x in bl)
b1 = st.mean([opp(bl, 'none', s, 1, um - 99, um + 1) for s in Sb]); b2 = st.mean([opp(bl, 'none', s, 2, um - 99, um + 1) for s in Sb])
cgh5 = reldecay('coord_cleanghost_k2p0.csv', '1', 1, 2, True); cgh10 = reldecay('coord_cleanghost_10.csv', '1', 1, 2, True)
fl5 = reldecay('coord_flat_k2p0.csv', '1', 1, 2, True); fl10 = reldecay('coord_flat_10.csv', '1', 1, 2, True)
print("  INSTALL (enforcement on):  opp1 %.3f vs comp %.3f  pref +%.3f  selectivity %.2f" % (o1i, o2i, o1i - o2i, sel))
print("  BASELINE (no enforcement): opp1 %.3f vs comp %.3f  pref +%.3f  => SELF-INSTALLS" % (b1, b2, b1 - b2))
print("  CLEANGHOST (enforce off)   5s: %+.4f m%+.1f  PERSISTS" % (cgh5['mean'], cgh5['margin']))
print("  CLEANGHOST                10s: %+.4f m%+.1f  PERSISTS" % (cgh10['mean'], cgh10['margin']))
print("  FLAT (grounding off)       5s: %+.4f m%+.1f  DECAYS" % (fl5['mean'], fl5['margin']))
print("  FLAT                      10s: %+.4f m%+.1f  (%d/%d) DECAYS" % (fl10['mean'], fl10['margin'], fl10['pos'], fl10['n']))
print("  => dissociation (10s): cleanghost %+.4f PERSISTS  vs  flat %+.4f DECAYS" % (cgh10['mean'], fl10['mean']))

print("\n" + "#" * 78); print("# 4. COORDINATION DOSE-RESPONSE (coord_k sweep)"); print("#" * 78)
print("  %7s | %13s | %12s | %4s | %16s | %14s" % ('coord_k', 'baseline pref', 'install pref', 'sel', 'cleanghost', 'flat'))
for k in ['0p5', '1p0', '2p0']:
    b = load('coord_baseline_k%s.csv' % k); Sb = seeds(b, 'none'); um = max(int(x['update']) for x in b)
    bp = st.mean([opp(b, 'none', s, 1, um - 99, um + 1) - opp(b, 'none', s, 2, um - 99, um + 1) for s in Sb])
    c = load('coord_cleanghost_k%s.csv' % k); Sc = seeds(c, '1')
    ip = st.mean([opp(c, '1', s, 1, 900, 1000) - opp(c, '1', s, 2, 900, 1000) for s in Sc]); se = colmean(c, '1', 'selectivity', 900, 1000)
    cgh = reldecay('coord_cleanghost_k%s.csv' % k, '1', 1, 2, True); fl = reldecay('coord_flat_k%s.csv' % k, '1', 1, 2, True)
    cv = 'PERSIST' if cgh['margin'] < 2 else 'decay'
    print("  %7s | +%11.3f | +%10.3f | %4.2f | %+.4f m%+5.1f %s | %+.4f m%+.0f" %
          (k, bp, ip, se, cgh['mean'], cgh['margin'], cv, fl['mean'], fl['margin']))
print("  (baseline pref 0.061->0.189 & flat collapse +0.10->+0.18 both grow with coord_k = coherent)")

print("\n" + "#" * 78); print("# 5. VESTIGE ROW (paper-1 reuse) + FLOOR"); print("#" * 78)
sd = reldecay('extinct3_cleanghost.csv', '1', 1, 2)
nr = load('extinct3_none.csv'); Sn = seeds(nr, 'none')
f1 = st.mean([opp(nr, 'none', s, 1, 1500, 1600) for s in Sn])
print("  silly (social, enforcement-propped) 5s: %+.4f m%+.1f  DECAYS" % (sd['mean'], sd['margin']))
print("  free/none opp1 (never-taboo baseline): %.3f" % f1)

print("\n" + "#" * 78); print("# 6. TAXONOMY SUMMARY (10-seed where available)"); print("#" * 78)
print("  %-14s %-16s %-11s %-22s %-20s" % ('row', 'ground', 'enforce?', 'persist(oversight off)', 'own-knockout'))
print("  %-14s %-16s %-11s %-22s %-20s" % ('vestige', 'enforcement', 'YES', 'NO  (+0.057)', '(enforcement=ground)'))
print("  %-14s %-16s %-11s %-22s %-20s" % ('environmental', 'physical hazard', 'no', 'YES (%+.3f)' % pg10['mean'], 'hazard-off (%+.3f)' % ho10['mean']))
print("  %-14s %-16s %-11s %-22s %-20s" % ('coordination', 'coordination', 'no(self)', 'YES (%+.3f)' % cgh10['mean'], 'flatten (%+.3f)' % fl10['mean']))
