# Internalization probe — pre-registration
**Committed 2026-08-11, before any probe run.** Predictions and kill conditions are fixed
here so the analysis cannot exercise post-hoc degrees of freedom. Any deviation from this
document is disclosed as a deviation.

## Question
Are the *surviving* norms **stored in the policy (internalized)** or **reconstructed each
step from the visible cue**? The probe runs on the norms that persist under oversight
removal — the **grounded** rows (environmental/poison, coordination) — not merely as a
training control. The vestige (silly), which does not survive, is the negative reference.

## Arm 1 — Frozen-weights (cheap, corroborating)
- **Manipulation:** freeze the policy at the switch (`freeze_after = 1000`); roll out the
  ghost phase with no further learning.
- **Prediction (fixed):** grounded norms hold under frozen weights → the behavior is in the
  weights, not re-learned each ghost step.
- **Interpretation limit (stated up front):** the ground is still live in the reward
  stream, so this is *consistent-with-stored*, **not proof**. Role: corroboration only.

## Arm 2 — Cue-masked (behavioral, the clean discriminator)
- **Manipulation:** mask the enforcement-relevant cue — the mark channel (mark planes +
  self-mark feature) — during the ghost phase (`unmark_after = 1000`), so the agent cannot
  re-derive "this is a policed berry" from observation.
- **Prediction (fixed):** if a grounded norm **survives** with the cue gone → it cannot be
  reconstructed from observation → **stored**. If it **collapses** → it was cue-driven.
- **Role:** THE behavioral discriminator between stored and reconstructed.

## Arm 3 — Hidden-state decode + ablate (flagship)
- **Manipulation:** linear-probe the GRU hidden state for the norm direction ("will
  avoid / will prefer berry X") through the ghost phase; then ablate the decoded direction
  from the carry during rollout and measure behavior change.
- **Predictions (fixed):**
  1. On **grounded** norms the direction is **decodable and stable** across the ghost phase.
  2. On the **vestige** it **decays toward chance**, tracking the behavioral reversion.
  3. **Ablating** the direction **moves behavior on the grounded norms** and **does not**
     on the reverted vestige.
- **Interpretation:** internalization shown as **represented + causal + located**. The
  contrast makes "discipline without the disciplined" rigorous: the substrate demonstrably
  *can* internalize (grounded norms carry a stable, causal representation), and the purely-
  enforced rule *specifically failed to* — its representation decayed with its behavior.

## KILL CONDITIONS (pre-committed)
- If the grounded-norm direction is **not decodable** → the stored/internalized claim FAILS;
  the title softens to the behavioral taxonomy only.
- If **ablation does not move behavior** on the grounded norms → same; representation is not
  shown causal, claim FAILS.
- If a grounded norm **collapses under cue-masking** (Arm 2) → it was cue-driven, not stored;
  that row is reported as reconstructed, not internalized.

## Build status (2026-08-11)
- **Arm 1 (frozen):** TURNKEY — `freeze_after` flag exists, oracle-clean, default-off.
- **Arm 2 (cue-masked):** TURNKEY — `unmark_after`/`mask_marks` exists (masks mark planes +
  self-mark feature in `observe`), oracle-clean, default-off.
- **Arm 3 (decode+ablate):** BUILD REQUIRED — `train_jax` carries the GRU state but never
  emits it. Needs (a) hidden-state logging through the ghost phase, (b) an offline linear
  probe, (c) a carry-intervention path to ablate the decoded direction mid-rollout.

## Scope fence
Runs on grounded norms (poison, coordination) at the flagship config; vestige as negative
reference. Non-blocking for the taxonomy submission; this is the NMI-ceiling layer.

---

# Appendix — Arm-3 refined predictions (committed 2026-08-11, AFTER Wave-1)
**These refine the original Arm-3 predictions above using the Wave-1 frozen/cue-mask
results. They are data-informed and dated as such — they are NOT presented as the
original pre-Wave-1 predictions. They are fixed here before any Arm-3 (decode/ablate) run.**

## Wave-1 empirical basis (flagship config, 5 seeds, ghost-end endpoint gap)
- **coordination cue-mask: +0.210 (103% of normal +0.203)** — SURVIVES fully.
- **poison cue-mask: +0.107 (46% of normal +0.234)** — HALVED, not collapsed; the surviving
  +0.107 is ~10× the vestige (still avoiding), tight (sd 0.013).
- **poison frozen: +0.241 (103%)** — holds in weights.
- (coord frozen pending; predicted ~100%.)

## Refined interpretation (fixed)
The only difference between poison-normal (+0.234) and poison-cue-mask (+0.107) is mark
visibility, so **poison persistence is ~half hazard-grounded (stored) and ~half mark-cue-
scaffolded (reconstructed from observation each step).** Coordination is **cue-independent
(cleanly internalized).** This is a dissociation *within* the grounded norms.

## Arm-3 predictions, refined (fixed before the decode/ablate run)
1. **Coordination:** the norm direction is decodable from the GRU state and **stable across
   the ghost phase, cue-independent**; ablating it **moves behavior**. (Clean internalization.)
2. **Poison:** the direction is decodable but **partly tied to the mark input** — expect a
   hazard-grounded component plus a mark-conditioned component. Ablating the stored direction
   moves the **cue-independent half** (~the +0.107 hazard floor), not the mark-scaffolded half.
3. **Vestige (reference):** decodability **decays toward chance** tracking behavioral
   reversion; ablation **does not move** behavior (nothing left inside).

## Kill conditions, refined (pre-committed)
- If the **coordination** direction is not decodable, or ablation does not move it → the
  clean-internalization claim FAILS; title softens to the behavioral taxonomy.
- If **poison** shows no decodable stored component (all variance is mark-conditioned) →
  poison is reported as **cue-scaffolded, not internalized** (consistent with its 46%
  cue-mask survival), and the internalization claim is carried by coordination alone.

## Addendum — coord-frozen result + dual-signature coordination prediction (2026-08-11)
`probe_coord_frozen`: ghost-end convergence gap **+0.158 (sd 0.005, n=5) = ~76% of install**,
holds ~16× above the vestige (does NOT collapse) but is a **partial hold, not the ~100%
originally predicted** (poison-frozen was 103%). Freeze verified working (train_jax:213-215
gate params; and the zap rise is evidence *for* freeze — a re-learning policy would unlearn
the now-unrewarded zapping, but frozen zaps *rose*). The zap rise (37→132) is downstream of
the convergence slip: freezing prevents re-convergence → more non-conformers → the frozen
enforcer-policy zaps them more (confirmed by contrast: poison held 103% → prevalence flat
0.069→0.070 → zaps barely move 9.7→14.7; coordination slipped → prevalence rose → zaps
tripled). Coordination is thus **cue-independent (cue-mask 103%) but only partly freeze-
stable (76%) — a mutual equilibrium partly sustained by ongoing re-adaptation.**

**Refined Arm-3 coordination prediction (fixed, must hold BOTH signatures):** the convergence
direction is **decodable and cue-independent** (consistent with cue-mask 103%), AND **shows
adaptation-sensitivity** (consistent with frozen 76%) — under ablation, behavior moves, and
the direction may be less stable / partly re-established through the ghost than poison's,
reflecting that coordination is re-adaptation-maintained, not purely weight-stored. If the
coordination direction is neither decodable nor causal, the clean-internalization claim
fails (as above); a decodable-but-adaptation-sensitive direction *confirms* the two-signature
reading (stored-and-cue-scaffolded [poison] vs cue-independent-and-adaptation-sustained [coord]).
