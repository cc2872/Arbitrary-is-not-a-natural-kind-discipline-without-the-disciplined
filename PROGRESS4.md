# Norm persistence under oversight removal — progress update
## 2026-08-08 (manuscript assembly + 2×2 completion + wrap-up; follows PROGRESS3.md)

PROGRESS3 established the flagship: under valid (full) oversight removal, the purely
social norm decays while the physically anchored one decays far less — a seed-robust
decay-rate dissociation. Since then the work has been (a) a new headline-upgrading
result, the enforcer-incentive-only cell; (b) full manuscript assembly into a PNAS
draft with figures; (c) an end-to-end audit that resolved every open `[CHECK]`/`[FILL]`
that is computable from data or code. This doc captures all of that and the remaining
path to submission.

### 1. NEW RESULT — the enforcer-incentive-only cell completes the 2×2 (the headline upgrade)
The central claim ("the norm rests on the enforcer's incentive, not the violator's
cost") previously rested on a two-cell inference (partial vs full removal). We built
an **independent bonus gate** (`gate_bonus_after`, decoupled from the violator-cost
`enforce` flag) and ran the missing fourth cell: keep the violator's zap penalty fully
armed, withhold only the enforcer's bonus at the switch. `extinct3_enforceronly.csv`,
10 seeds, gate verified (enforce on throughout; `bonus_on` 1→0 at update 1000).

| removal | silly (1,) | poison (0,) |
|---|---|---|
| violator cost only (partial) | −0.004 (−0.3) HOLDS | +0.000 (0.0) |
| **enforcer incentive only** | **+0.062 (margin 35.7; 10/10)** DECAYS | +0.024 (2.9; 9/10) |
| full (both channels) | +0.057 (14.7) | +0.016 (2.2; 7/10) |

**Gating the enforcer's incentive alone dissolves the norm as completely as full
removal (silly +0.062 vs +0.057); gating the violator's cost alone does nothing.**
Enforcer-only ≈ full ⇒ the no-interaction assumption behind the old inference holds;
the bonus channel is sufficient, the violator channel neither necessary nor sufficient.
**Smoking gun:** in this arm the zap penalty stays armed yet zaps collapse **43.7 → 0.7
/episode** once the bonus is withheld — no one enforces a rule that no longer pays, so
the violator meets no enforcement and the norm decays (opp₁ 0.239 → 0.470). The
headline is now a **demonstrated 2×2**, not an inference.

### 2. Window unification (the critical-path recompute)
Two windows were in play (install/dose last-10%; extinction last-5%/80; text said
"final 100"). Recomputed EVERYTHING at one window — **last 100 updates** (install-only:
1400–1499; two-phase: install-end 900–999, gate-end 1500–1599). Invariant: Tables 1–3,
silly decay +0.057 / 85%, endpoint opp₁ 0.468, residual +0.0099. **Changed** (was
last-80): full-poison decay +0.021→**+0.016** (margin 2.74→2.2); paired differential
+0.042→**+0.046** (margin 4.8, 5/5 same sign); absolute fold 2.7×→**3.4×**; fractional
**85% vs 7%** (each vs its own berry-2-differenced install gap — the only apples-to-
apples pair; the 0.187/0.093 denominators were dropped as the source of the earlier
7%-vs-10-15% conflict). Abstract "2.7-fold" → "roughly 3-fold" (window-robust).

### 3. Audit results — every computable CHECK/FILL resolved
- **Self-mark visibility ERROR (fixed):** paper said marks are "visible to other agents
  only"; the code shows the agent observes its OWN mark (`berryworld_jax.py:139`
  self-feature + `:119` mk-plane centre). Corrected to "visible to self and others";
  Köster-divergence flagged `[VERIFY vs their SI]`, not asserted.
- **enc₁ starvation check (null is informative):** partial-removal ghost berry-1
  encounters 548.6/ep vs full 392.3 (none 389.5). Agents meet berry-1 abundantly and
  still decline it — "making violation free changed nothing" is safe, not starvation.
- **Reverse-sequencing (free result, HOLDS):** under full removal, enforcement collapses
  within ~50 updates (zaps 44 → <8 by upd 40, ~1 by 150) while compliance relaxes over
  the full 600 (opp₁ 0.33→0.47). Enforcement extinguishes before compliance decays — the
  mirror of the acquisition-time ordering. Panel built (`fig_sequencing`).
- **Gate-1 stats:** return −1.78→+0.40→+14.48; sel 1.13 / 1.04; eat0 91→67.5→45.3; zaps
  3→58→67 (none/(0,)/(0,1), 2-berry N=12).
- **Poison absolutes pinned:** paper's 0.181→0.267 = the 10-seed run (0.183→0.266);
  PROGRESS3's 0.175→0.254 was the 5-seed. Two-phase install-end sits at 0.183 (not
  Table 2's 0.123) because install is 1000 updates vs 1500 — expected, not an error.
- **Definitions:** margin = mean/SE across seeds (one-sample t); selectivity =
  P(target marked | zap) ÷ P(marked) = (zaps-on-marked/landed)/(marked/active), >1 =
  targeting above base rate; d-i-d = `(oppₘ^none−oppₘ^cond)−(opp_c^none−opp_c^cond)`,
  CRN-paired within seed (conditions share PRNGKey(0)), positive = berry-specific.
- **Unmarked-zap semantics:** zapping unmarked = zapper −c_zap (0.1, forgone bonus),
  target −c_zapped (2) — "pure loss to both" is accurate; NO explicit extra penalty for
  zapping the innocent (weaker than Köster's ~−20; note it).
- **Methods constants (from code):** grid 15 (2-berry)/22 (3-berry); N=12 (8 density);
  view 3 → 7×7 window; 7 actions (4 move, eat, zap, no-op); r_eat +1, delayed r_poison
  −4 at D=100, c_zap 0.1, c_zapped 2 (sweep 2/15/35), bonus b 8.75 (sweep 2/4/8.75),
  mark_steps 40, zap_range 4, removal 25, regrow 0.01; Adam lr 3e-4 under global-norm
  clip 0.5, γ 0.99, GAE λ 0.95, clip 0.2, 3 epochs, entropy 0.01, vf 0.5, hidden 64,
  num_envs 256; episode 300 steps → 76,800 env-steps/update; install 1500, two-phase
  1000+600. Oracle diff: 20 deterministic steps × all conditions × 5 seeds × 11
  quantities, max abs deviation 0.0.

### 4. Figures (built + rendered, self-checked at final print width, vector, fonts embedded)
- **Fig 2 `fig2_install`** — dose-response: silly install + selectivity vs bonus; install
  tracks enforcement coming online (~0 where sel<1).
- **Fig 3 = `fig4_dissociation` (left) + `F5_recovery` (right)** — decay-rate dissociation
  curves + biphasic recovery to baseline (85% erased, residual still declining).
- **`fig_ghosts`** — the removal contrast as a net-decay effect summary (extend to the
  full 2×2 now that enforcer-only is in: partial / enforcer-only / full).
- **`fig_sequencing`** — reverse-ordering panel (enforcement collapses before compliance).
- Style module `pnas_style.py` (single-col 8.7 / 1.5-col 11.4 / full 17.8 cm; Adam-free,
  fonts embedded). Lesson banked: size at final width, render a PNG and eyeball before
  handing over; keep all labels inside axis limits so `constrained_layout` never clips.

### 5. Manuscript state
Full PNAS draft compiles to 6 pages (3 figures + 4 tables): methods confound →
opportunity metric → affordable install + dose-response → two-channel removal →
decay-rate dissociation → rational-abandonment reading. Framing calibrated: "decay-RATE
dissociation" (not "poison frozen"), model-organism register with disanalogies stated,
directions robust / magnitudes configuration-dependent. Title in play; abstract +
significance assembled.

### 6. Confirmatory runs (the near-zero-marginal-cost hardening)
Infra + notebooks built; enforcer-only DONE (§1). Still pending (not load-bearing;
queue when pods free):
- **`extinct3_seeds20`** — full removal at 20 seeds; tightens the load-bearing contrast
  against the "add seeds" revision ask.
- **`extinct3_longtail`** — 2000-update ghost phase; tests whether the silly residual
  reaches baseline (closes "completion not proven at this horizon").

---

## WRAP-UP STAGE — remaining path to submission

**Author-only (cannot be done from data/code):**
1. **Fig 1** environment schematic (illustration; adapt Köster Fig 1's three-panel).
2. **Refs 3 & 4** real lookup (Vinitsky, Collect. Intell. volume/pages; Gelpí, PNAS
   Nexus fields) — currently `[VERIFY]`.
3. **Köster self-mark SI check** — confirm whether the original hides self-marks before
   asserting the divergence (§3).
4. **Acknowledgments** — GPU provider for the 80 GB runs (+ university HPC allocation if used).

**Ready to paste (values/text in hand):**
5. Unify the endpoint window to last-100 and swap the changed numbers (§2): Table 4
   poison +0.016, differential +0.046, fold "roughly 3-fold", fractional "85% vs 7%".
6. Add the enforcer-only row to Table 4 and upgrade the Discussion "lives in the
   enforcer's incentive" sentence from inference to demonstration (§1) — paste-ready.
7. Fill Methods `[FILL]`s from §3 constants; convert the three definition `[CHECK]`s.
8. Fix the self-mark sentence + Methods `[CHECK]` (§3).
9. `norm_paper.bib` — draft the resolvable entries (Köster, Schelling, Bicchieri,
   Hadfield-Menell, Schulman/PPO, JAX, DeepMind-JAX); stub the 2 `[VERIFY]`.

**Optional hardening (queue if pods free; not blocking):**
10. `seeds20` (power), `longtail` (tail completion); fold enforcer-only into `fig_ghosts`
    as a third x-group so the 2×2 reads visually.

**Assembly:** figure-file names (fig4/F5) render as paper Fig 3 — clean up numbering at
final assembly.

*Stack: JAX/Flax/Optax recurrent IPPO. Repro: env oracle-diffed 0.00e+00; all flags
(`enforce`, `ghost_keeps_bonus`, `freeze_after`, `isolate_after`/`n_focal`,
`unmark_after`, `gate_bonus_after`) default-off and bit-exact. Extinction driver runs
seeds sequentially (memory-safe); notebooks set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Data: extinct3_cleanghost(.csv)/_poison10 (full removal), extinct3.csv (partial),
extinct3_enforceronly.csv (2×2 fourth cell), extinct3_isolate.csv (coordination).*
