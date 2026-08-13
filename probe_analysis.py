"""probe_analysis.py -- CRN-paired read for the perception-probe arms (M1 self-mask,
M2 self+world mask) vs their unmasked control, on the violator-gated (or, for the
placebo, enforcer-gated) ghost. Everything is measured on the SILLY berry (cond '1',
berry 1) as an opportunity-controlled per-encounter rate opp1 = sum(eat1)/sum(enc1).

The question the mask adjudicates is reading (c) -- reconstruction. In the violator-
gated ghost the mark cue is still live (eating berry-1 sets self_mark; others still
see your mark plane), so berry-1 avoidance could be re-derived each step from that
cue rather than stored. The masks sever it: M1 the self channel, M2 additionally the
world channel. Scope: this touches PERCEPTION only, not the reward landscape, so a
clean HOLD kills (c) but leaves (a) flat-gradient open (that's the mark-cue #4 mask +
the standing-stock re-run). Do not read a HOLD as "null explained".

Pre-registered verdict on masked ghost-end opp1 (unmasked violator ~0.068):
  HOLDS      masked within +0.02 of unmasked      -> perceived enforcement removed,
                                                     rule holds anyway -> (c) dead for
                                                     that channel.
  CONFIRMS   masked rises to the vestige-decay     -> removing perceived (costless)
             band (>=~0.35, toward 0.35-0.47)        enforcement dissolves the rule
                                                     -> (c) confirmed (Durkheim: the
                                                     rule was the perceived cue).
  PARTIAL    in between (>0.088, <0.35)            -> partial dependence; escalate to
                                                     20 seeds / extend the ghost.

Usage:
  import probe_analysis as P
  P.report('extinct3_m1_masked.csv', 'extinct3_m1_unmasked.csv', label='M1 (self-mask)')
  P.report('extinct3_m2_masked.csv', 'extinct3_m1_unmasked.csv', label='M2 (self+world)', surge_check=True)
  P.report('extinct3_placebo_m1.csv', 'extinct3_enforceronly.csv', label='PLACEBO M1 (enforcer arm)')
"""
import csv
import numpy as np

HOLD_EPS = 0.02          # masked within +HOLD_EPS of unmasked -> HOLDS
CONFIRM_LO = 0.35        # masked >= this -> CONFIRMS (vestige-decay band lower edge)


def _rows(path):
    return list(csv.DictReader(open(path)))


def _install(r):
    return (float(r["enforce"]) > 0.5 and float(r.get("bonus_on", 1) or 1) > 0.5
            and float(r.get("removal_on", 1) or 1) > 0.5
            and float(r.get("hazard_on", 1) or 1) > 0.5
            and float(r.get("coord_on", 1) or 1) > 0.5)


def _switch(rows):
    return max(int(r["update"]) for r in rows if _install(r))


def _seed_windows(rows, cond, window):
    ups = sorted({int(r["update"]) for r in rows})
    sw = _switch(rows)
    w0 = set(sorted(u for u in ups if u <= sw)[-window:])     # install-end
    w1 = set(sorted(u for u in ups if u > sw)[-window:])      # ghost-end
    return sw, w0, w1


def _rate(rows, cond, seed, wins, num, den):
    rs = [r for r in rows if r["condition"] == cond and int(r["seed"]) == seed
          and int(r["update"]) in wins]
    n = sum(float(r[num]) for r in rs)
    d = sum(float(r[den]) for r in rs)
    return n / max(d, 1e-9)


def _mean_col(rows, cond, seed, wins, col):
    rs = [r for r in rows if r["condition"] == cond and int(r["seed"]) == seed
          and int(r["update"]) in wins]
    return float(np.mean([float(r[col]) for r in rs])) if rs else float("nan")


def report(masked_csv, unmasked_csv, cond="1", berry=1, window=100,
           label="", surge_check=False, placebo=False):
    mk = _rows(masked_csv); un = _rows(unmasked_csv)
    seeds = sorted(set(int(r["seed"]) for r in mk if r["condition"] == cond)
                   & set(int(r["seed"]) for r in un if r["condition"] == cond))
    sw_m, w0_m, w1_m = _seed_windows(mk, cond, window)
    sw_u, w0_u, w1_u = _seed_windows(un, cond, window)
    e, c = f"eat{berry}", f"enc{berry}"
    print("=" * 84)
    print(f"PERCEPTION PROBE  {label}")
    print(f"  masked   = {masked_csv}   (switch@{sw_m})")
    print(f"  unmasked = {unmasked_csv} (switch@{sw_u})")
    print(f"  paired seeds = {seeds}   berry={berry} (silly)   cond='{cond}'  window=last {window}")
    print("=" * 84)

    # --- (1) install-phase bit-identity: masked==unmasked BEFORE the switch (same keys,
    #     mask off until the switch) -> every logged quantity must match at install-end.
    ident_cols = [e, c, "zaps", "prevalence", "ret"]
    bad = []
    for s in seeds:
        for col in ident_cols:
            vm = _mean_col(mk, cond, s, w0_m, col); vu = _mean_col(un, cond, s, w0_u, col)
            if not np.isclose(vm, vu, rtol=0, atol=1e-6):
                bad.append((s, col, vm, vu))
    if bad:
        print("  !! INSTALL BIT-IDENTITY FAILED (masked and unmasked diverge PRE-switch):")
        for s, col, vm, vu in bad[:6]:
            print(f"       seed {s} {col}: masked {vm:.6f} != unmasked {vu:.6f}")
        print("     -> keys/config differ between the two runs; pairing is INVALID.")
    else:
        print(f"  install bit-identity OK: all {ident_cols} match pre-switch across {len(seeds)} seeds")

    # --- (2) paired ghost-end opp1 (the verdict)
    opp_m = np.array([_rate(mk, cond, s, w1_m, e, c) for s in seeds])
    opp_u = np.array([_rate(un, cond, s, w1_u, e, c) for s in seeds])
    d = opp_m - opp_u
    margin = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    print(f"\n  opp1 ghost-end   unmasked {opp_u.mean():.3f}   masked {opp_m.mean():.3f}"
          f"   paired delta {d.mean():+.3f}  margin {margin:+.2f} ({len(seeds)} seeds)")

    # --- (3) sampling-live guard (enc1 must stay high or the null is an under-sampling artifact)
    enc_m = np.array([_mean_col(mk, cond, s, w1_m, c) for s in seeds])
    print(f"  enc1 ghost-end   masked {enc_m.mean():.1f}/ep   "
          f"{'OK (>100, sampling live)' if enc_m.mean() > 100 else '!! <100 -> UNDER-SAMPLING, null invalid'}")

    # --- (4) mark prevalence (regime check)
    pv_m = np.array([_mean_col(mk, cond, s, w1_m, "prevalence") for s in seeds])
    pv_u = np.array([_mean_col(un, cond, s, w1_u, "prevalence") for s in seeds])
    print(f"  prevalence ghost-end  unmasked {pv_u.mean():.3f}   masked {pv_m.mean():.3f}")

    # --- (5) surge check (M2 only: masking world marks can break enforcer targeting)
    zap_m = np.array([_mean_col(mk, cond, s, w1_m, "zaps") for s in seeds])
    zap_u = np.array([_mean_col(un, cond, s, w1_u, "zaps") for s in seeds])
    print(f"  zaps ghost-end   unmasked {zap_u.mean():.0f}/ep   masked {zap_m.mean():.0f}/ep")
    if surge_check:
        ratio = zap_m.mean() / max(zap_u.mean(), 1e-9)
        ok = zap_m.mean() >= 400 and ratio >= 0.6
        print(f"    SURGE CHECK: masked/unmasked zap ratio {ratio:.2f}  "
              f"{'OK (surge preserved -> clean perception test)' if ok else '!! SURGE COLLAPSED -> M2 confounded occurrence with perception; fall back to M1'}")

    # --- (6) verdict
    mg = opp_m.mean()
    if placebo:
        # inverted: the mask is applied where the DECAY is driven by something else
        # (enforcer-incentive removal, not perception). It should be behaviorally INERT
        # -> masked decay must MATCH the unmasked decay. A significant delta means the
        # mask op itself perturbs dynamics, so a violator-arm collapse would be confounded.
        inert = abs(d.mean()) <= HOLD_EPS and (np.isnan(margin) or abs(margin) < 2)
        verdict = ("MASK INERT (placebo PASSES: self-mask did not change the enforcer-arm "
                   "decay -> clean scalpel; a violator-arm collapse = removed perception)"
                   if inert else
                   "MASK PERTURBS (placebo FAILS: masking self-perception itself shifts the "
                   "decay -> a violator-arm collapse would be confounded by the op)")
        print(f"\n  VERDICT: {verdict}")
        print("  NOTE: self_mark is NOT near-empty in the enforcer arm (agents forage -> get"
              " marked, prevalence ~0.55), so this is a STRONG inertness test, not a no-op.")
        return dict(opp_masked=mg, opp_unmasked=float(opp_u.mean()), delta=float(d.mean()),
                    margin=float(margin), enc1=float(enc_m.mean()), verdict=verdict)
    if mg <= opp_u.mean() + HOLD_EPS:
        verdict = "HOLDS  -> (c) dead for this channel (perceived enforcement removed, rule holds)"
        escalate = False
    elif mg >= CONFIRM_LO:
        verdict = "CONFIRMS -> (c) confirmed (removing perceived enforcement dissolves the rule)"
        escalate = False
    else:
        verdict = "PARTIAL -> partial dependence"
        escalate = True
    print(f"\n  VERDICT: {verdict}")
    if escalate:
        print(f"    ghost-end opp1 {mg:.3f} in the ambiguous zone (>{opp_u.mean()+HOLD_EPS:.3f}, <{CONFIRM_LO})"
              f" -> PRE-REGISTERED ESCALATION to 20 seeds / extend ghost.")
    print("  SCOPE: adjudicates reading (c) only. A HOLD leaves (a) flat-gradient OPEN"
          " (that's the mark-cue mask + standing-stock re-run). Not 'null explained'.")
    return dict(opp_masked=mg, opp_unmasked=float(opp_u.mean()), delta=float(d.mean()),
                margin=float(margin), enc1=float(enc_m.mean()), verdict=verdict)


if __name__ == "__main__":
    import sys
    report(sys.argv[1], sys.argv[2], label=sys.argv[3] if len(sys.argv) > 3 else "")
