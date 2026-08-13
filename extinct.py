"""
extinct.py -- two-phase extinction driver. Install a norm under enforcement, then
remove enforcement (ghost) and watch whether the acquired avoidance DECAYS.

Uses train_jax.make_train(cfg, pm, hp, n_install=K, freeze_after=F): enforce=True
for updates < K (install), enforce=False after (extinction/ghost). If F is set,
weights also freeze at update F (skip optimizer update, behavior still logged) --
the "stored not reconstructed" control that subtracts continued-training drift.
Sequential seeds (memory-safe). Generalized to n_berry_types >= 2 (reads the env's
n_berry_types / grid), so the 3-berry world (0=poison, 1=silly, 2=harmless alt) --
the one where the silly rule actually installs -- runs here.

The read: within a condition, opp_t = eat_t/enc_t per update. Install phase drives
it down (compliance); after the enforce=False switch, does it climb back
(extinction) or hold (persistence)? Poison (intrinsic penalty) should persist; a
purely-social silly-rule norm is the one whose fate is the flagship question. The
learning-on arm (freeze_after=None) can re-equilibrate under the shifted reward
landscape; the FROZEN arm (freeze_after=n_install) is the clean persistence test.

Run 2-berry:   python extinct.py --run
Run 3-berry:   python extinct.py --run3            (learning-on)
               python extinct.py --frozen3         (frozen-weights control)
Analyze:       python extinct.py --analyze extinct3.csv
"""
import argparse
import csv
import jax
import numpy as np
import run_sweep as R
import train_jax as T
import berryworld_jax as bwj


def run_extinction(conditions, env, hp, n_install, n_seeds=5, N=12,
                   out_csv="extinct.csv", freeze_after=None,
                   isolate_after=None, n_focal=1, unmark_after=None,
                   gate_bonus_after=None, gate_removal_after=None,
                   hazard_off_after=None, flatten_returns_after=None,
                   mask_self_after=None, vmap_seeds=False):
    if not any(d.platform == "gpu" for d in jax.devices()) \
            and hp["updates"] * hp["num_envs"] > 1000:
        raise RuntimeError(f"extinct aborted: NO GPU ({jax.devices()}); this is a "
                           f"real run (updates*num_envs={hp['updates']*hp['num_envs']}).")
    nbt = env.get("n_berry_types", 2)
    grid = env.get("grid", 15)
    fields = (["condition", "seed", "update", "enforce", "freeze", "bonus_on", "removal_on",
               "hazard_on", "coord_on", "self_cue_on"]
              + [f"eat{t}" for t in range(nbt)] + [f"enc{t}" for t in range(nbt)]
              + ["selectivity", "prevalence", "zaps", "ret"])
    f = open(out_csv, "w", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()
    try:
        for marked in conditions:
            cfg = bwj.JCfg(
                n_agents=N, episode_len=env["episode_len"],
                poison_delay=env["poison_delay"], zap_removal_steps=env["zap_removal_steps"],
                r_zap_bonus=env["r_zap_bonus"],
                observe_pending=env.get("observe_pending", False),
                bonus_requires_mark=env.get("bonus_requires_mark", False),
                auto_target=env.get("auto_target", False),
                c_zapped=env.get("c_zapped", 2.0),
                grid=grid, n_berry_types=nbt,
                ghost_keeps_bonus=env.get("ghost_keeps_bonus", True),
                convergent_berry=env.get("convergent_berry", None),
                coord_k=env.get("coord_k", 0.0), coord_a=env.get("coord_a", 1.5),
                conformity_berry=env.get("conformity_berry", None),
                convention_berries=tuple(env.get("convention_berries", ())),
                marked_mask=tuple(t in marked for t in range(nbt)))
            pm = T.build_patch_mask(marked, N, n_berry_types=nbt, grid=grid)
            train1 = T.make_train(cfg, pm, hp, n_install=n_install,
                                  freeze_after=freeze_after,
                                  isolate_after=isolate_after, n_focal=n_focal,
                                  unmark_after=unmark_after, gate_bonus_after=gate_bonus_after,
                                  gate_removal_after=gate_removal_after,
                                  hazard_off_after=hazard_off_after,
                                  flatten_returns_after=flatten_returns_after,
                                  mask_self_after=mask_self_after)
            keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
            if vmap_seeds:
                # CHUNKED vmap: run `chunk` seeds per graph, loop the chunks. All 20 at
                # once needs ~n_seeds x the trajectory memory (~4GB/seed here -> 20 = 82GB,
                # OOMs an 80GB card); a chunk of 5 is ~20-35GB and fits with headroom while
                # still parallelizing. Same keys as the sequential path so seed s is
                # identical; XLA codegen differs (batched) -> verify against the known
                # sequential anchors (violator silly ~-0.004, enforcer ~+0.062). Pick a
                # chunk that divides n_seeds (else the remainder recompiles). True -> 5.
                chunk = n_seeds if vmap_seeds is True and n_seeds <= 5 else \
                    (5 if vmap_seeds is True else int(vmap_seeds))
                vm = jax.jit(jax.vmap(lambda k: train1(k)[1]))
                # ADAPTIVE: on OOM (smaller/busier GPU), halve the chunk and retry down to
                # 1 seed, then keep the working size -> the same notebook runs on any card.
                parts = []; i = 0
                while i < n_seeds:
                    c = min(chunk, n_seeds - i)
                    while True:
                        try:
                            mc = jax.block_until_ready(vm(keys[i:i + c])); break
                        except Exception as e:
                            if "RESOURCE_EXHAUSTED" in str(e) and c > 1:
                                c = max(1, c // 2)
                                print(f"  OOM -> retry with chunk={c}", flush=True); continue
                            raise
                    parts.append({kk: np.asarray(vv) for kk, vv in mc.items()}); del mc
                    print(f"  vmap chunk seeds {i}..{i + c - 1} done (chunk {c})", flush=True)
                    chunk = c; i += c            # stick with the size that fit
                m = {kk: np.concatenate([p[kk] for p in parts], 0) for kk in parts[0]}
            else:
                seed_metrics = jax.jit(lambda k: train1(k)[1])
                per_seed = []
                for k in keys:
                    mk = jax.block_until_ready(seed_metrics(k))
                    per_seed.append({kk: np.asarray(vv) for kk, vv in mk.items()}); del mk
                m = {kk: np.stack([ps[kk] for ps in per_seed], 0) for kk in per_seed[0]}
            cond = "".join(str(b) for b in marked) or "none"
            U = m["eat0"].shape[1]
            for s in range(n_seeds):
                for u in range(U):
                    row = dict(condition=cond, seed=s, update=u,
                               enforce=float(m["enforce"][s, u]),
                               freeze=float(m["freeze"][s, u]) if "freeze" in m else 0.0,
                               bonus_on=float(m["bonus_on"][s, u]) if "bonus_on" in m else 1.0,
                               removal_on=float(m["removal_on"][s, u]) if "removal_on" in m else 1.0,
                               hazard_on=float(m["hazard_on"][s, u]) if "hazard_on" in m else 1.0,
                               coord_on=float(m["coord_on"][s, u]) if "coord_on" in m else 1.0,
                               self_cue_on=float(m["self_cue_on"][s, u]) if "self_cue_on" in m else 1.0,
                               selectivity=float(m["selectivity"][s, u]),
                               prevalence=float(m["prevalence"][s, u]),
                               zaps=float(m["zaps"][s, u]), ret=float(m["ret"][s, u]))
                    for t in range(nbt):
                        row[f"eat{t}"] = float(m[f"eat{t}"][s, u])
                        row[f"enc{t}"] = float(m[f"enc{t}"][s, u])
                    w.writerow(row)
            f.flush()
            e = m["eat0"]; print(f"cond {cond:4s} done: eat0 {e[:, :5].mean():.0f}->{e[:, -5:].mean():.0f}"
                                 f"  (freeze={freeze_after} isolate={isolate_after} "
                                 f"n_focal={n_focal} unmark={unmark_after} "
                                 f"ghost_keeps_bonus={env.get('ghost_keeps_bonus', True)})", flush=True)
    finally:
        f.close()
    print(f"wrote -> {out_csv}", flush=True)


def analyze(csv_path, window=100, converge=()):   # window = last-N updates (canonical 100)
    """Decay read, RELATIVE frame. When enforcement lifts, general caution relaxes
    and ALL eat-rates rise -- so a marked berry's 'decay' must be measured against
    an UNMARKED control berry's drift. Control = the highest-index berry NOT marked
    in that condition (the harmless alternative berry in the 3-berry world; the other
    berry in the 2-berry world). Per marked berry m: rel_decay =
    (opp_m[ext-end] - opp_m[install-end]) - (same for the control berry), paired over
    seeds. >0 & margin>=2 = decayed BEYOND drift (extinction); ~0 = persisted.
    converge = berry indices whose norm is CONVERGENCE (coordination row): the norm is
    opp_C HIGH, so decay = opp_C FALLS. For those the sign is flipped so >0 still = decay
    (convergence lost), keeping the verdict semantics identical across avoidance/convergence."""
    converge = set(converge)
    rows = list(csv.DictReader(open(csv_path)))
    nbt = sum(1 for k in rows[0] if k.startswith("eat"))
    ups = sorted({int(r["update"]) for r in rows})
    # install phase = both channels live (enforce AND enforcer bonus); the switch is the
    # last such update. Works for all cells: violator-only (enforce drops), enforcer-only
    # (bonus_on drops), full (both drop). Old CSVs without bonus_on default it to 1.
    def _install(r):
        return (float(r["enforce"]) > 0.5 and float(r.get("bonus_on", 1)) > 0.5
                and float(r.get("removal_on", 1)) > 0.5
                and float(r.get("hazard_on", 1)) > 0.5 and float(r.get("coord_on", 1)) > 0.5)
    sw = max((int(r["update"]) for r in rows if _install(r)), default=-1)
    umax = ups[-1]
    w0 = sorted(u for u in ups if u <= sw)[-window:]   # install-end (last `window` updates)
    w1 = sorted(u for u in ups if u > sw)[-window:]    # extinction-end (last `window` updates)
    frozen = any(float(r.get("freeze", 0)) > 0.5 for r in rows)
    print(f"install: 0..{sw}   extinction: {sw+1}..{umax}   "
          f"[{'FROZEN weights' if frozen else 'learning-on'}]   (windows: end-of-each)\n")

    def opp_win(cond, seed, i, win):
        rs = [r for r in rows if r["condition"] == cond and int(r["seed"]) == seed
              and int(r["update"]) in win]
        e = sum(float(r[f"eat{i}"]) for r in rs); c = sum(float(r[f"enc{i}"]) for r in rs)
        return e / max(c, 1e-9)

    conds = sorted({r["condition"] for r in rows}); seeds = sorted({int(r["seed"]) for r in rows})
    tag = {0: "poison(physical->persist?)", 1: "silly(social->decay?)"}
    print("norm = marked-berry avoidance; rel_decay = its rise MINUS an unmarked berry's drift")
    print(f"{'cond':>5} {'mark':>4} | {'opp_marked inst->ext':>22} | {'rel_decay':>10} {'margin':>7} | verdict")
    for c in conds:
        if c == "none":
            continue
        marked_set = {int(ch) for ch in c}
        unmarked = [t for t in range(nbt) if t not in marked_set]
        if not unmarked:
            continue
        ctrl = max(unmarked)                         # harmless alt in 3-berry; other berry in 2-berry
        for mi in sorted(marked_set):
            sign = -1.0 if mi in converge else 1.0   # convergence: decay = opp_C FALLS
            rel = []; mi0 = mi1 = 0.0; n = 0
            for s in seeds:
                dm = opp_win(c, s, mi, w1) - opp_win(c, s, mi, w0)
                du = opp_win(c, s, ctrl, w1) - opp_win(c, s, ctrl, w0)
                rel.append(sign * (dm - du))
                mi0 += opp_win(c, s, mi, w0); mi1 += opp_win(c, s, mi, w1); n += 1
            rel = np.array(rel)
            margin = rel.mean() / (rel.std(ddof=1) / np.sqrt(len(rel))) if len(rel) > 1 else float("nan")
            if margin >= 2:
                verdict = "DECAYS"
            elif abs(margin) < 2:
                verdict = "PERSISTS"
            else:                                    # margin <= -2
                verdict = "PERSISTS(+)" if mi in converge else "?"   # convergence strengthened
            kind = "coord(converge->persist?)" if mi in converge else tag.get(mi, '')
            print(f"{c:>5} {mi:>4} | {mi0/n:.3f} -> {mi1/n:.3f}          "
                  f"| {rel.mean():+.4f} {margin:+7.2f} | {verdict}  {kind}  (ctrl=berry{ctrl})")
    if converge:
        print("\n(converge mode: rel_decay > 0 = convergence LOST; <= 0 = held/strengthened.)")
    else:
        print("\nDecay read: rel_decay > 0 & margin >= 2 = decayed beyond drift; ~0 = persisted.")
    print("Learning-on can re-equilibrate; confirm the frozen arm where a stored trace is claimed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="2-berry learning-on")
    ap.add_argument("--run3", action="store_true", help="3-berry learning-on")
    ap.add_argument("--frozen3", action="store_true", help="3-berry frozen-weights control")
    ap.add_argument("--isolate3", action="store_true", help="3-berry coordination knockout")
    ap.add_argument("--cleanghost3", action="store_true", help="3-berry FULL oversight removal (bonus gated)")
    ap.add_argument("--enforceronly3", action="store_true", help="3-berry ENFORCER-incentive-only removal (violator cost kept)")
    ap.add_argument("--storage3", action="store_true", help="3-berry storage test (bonus gated+isolate+no-cue+frozen)")
    ap.add_argument("--smokebonus", action="store_true", help="tiny enforcer-only gate smoke")
    ap.add_argument("--timeoutonly3", action="store_true", help="3-berry: gate timeout only (penalty+bonus kept)")
    ap.add_argument("--penaltyonly3", action="store_true", help="3-berry: gate zap penalty only (timeout+bonus kept)")
    ap.add_argument("--smokeremoval", action="store_true", help="tiny timeout-gate smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke3", action="store_true")
    ap.add_argument("--smokeiso3", action="store_true")
    ap.add_argument("--analyze", metavar="CSV")
    a = ap.parse_args()
    ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,
                        zap_removal_steps=25, bonus_requires_mark=True)
    ENV3 = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,
                         zap_removal_steps=25, bonus_requires_mark=True,
                         c_zapped=2.0, n_berry_types=3, grid=22)
    if a.analyze:
        analyze(a.analyze)
    elif a.smoke:
        hp = dict(R.FAITHFUL_HP); hp["num_envs"] = 16; hp["updates"] = 30
        run_extinction([(0, 1)], ENV, hp, n_install=15, n_seeds=1, out_csv="extinct_smoke.csv")
        analyze("extinct_smoke.csv")
    elif a.smoke3:
        hp = dict(R.FAITHFUL_HP); hp["num_envs"] = 8; hp["updates"] = 20
        run_extinction([(1,)], ENV3, hp, n_install=10, n_seeds=1,
                       freeze_after=10, out_csv="extinct3_smoke.csv")
        analyze("extinct3_smoke.csv")
    elif a.smokeiso3:
        hp = dict(R.FAITHFUL_HP); hp["num_envs"] = 8; hp["updates"] = 20
        run_extinction([(1,)], ENV3, hp, n_install=10, n_seeds=1,
                       isolate_after=10, n_focal=1, out_csv="extinct3_iso_smoke.csv")
        analyze("extinct3_iso_smoke.csv")
    elif a.smokebonus:
        # enforcer-only: enforce never gated (n_install=updates), bonus gated at 10
        hp = dict(R.FAITHFUL_HP); hp["num_envs"] = 8; hp["updates"] = 20
        run_extinction([(1,)], ENV3, hp, n_install=20, n_seeds=1,
                       gate_bonus_after=10, out_csv="extinct3_bonus_smoke.csv")
        analyze("extinct3_bonus_smoke.csv")
    elif a.smokeremoval:
        # timeout-only: enforce (penalty) never gated, removal gated at 10
        hp = dict(R.FAITHFUL_HP); hp["num_envs"] = 8; hp["updates"] = 20
        run_extinction([(1,)], ENV3, hp, n_install=20, n_seeds=1,
                       gate_removal_after=10, out_csv="extinct3_removal_smoke.csv")
        analyze("extinct3_removal_smoke.csv")
    elif a.run:
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600   # ~1000 install + 600 extinction
        run_extinction([(), (0,), (1,)], ENV, hp, n_install=1000, out_csv="extinct.csv")
        analyze("extinct.csv")
    elif a.run3 or a.frozen3:
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        fa = 1000 if a.frozen3 else None
        out = "extinct3_frozen.csv" if a.frozen3 else "extinct3.csv"
        run_extinction([(), (0,), (1,)], ENV3, hp, n_install=1000,
                       freeze_after=fa, out_csv=out)
        analyze(out)
    elif a.isolate3:
        # coordination knockout: install with all 12 agents (coord on), then at the
        # switch deactivate all but n_focal (coord off). learning-on, so it pairs with
        # extinct3.csv (ghost, coord on). silly first so it can't be starved.
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        run_extinction([(1,), (0,)], ENV3, hp, n_install=1000,
                       isolate_after=1000, n_focal=1, out_csv="extinct3_isolate.csv")
        analyze("extinct3_isolate.csv")
    elif a.cleanghost3:
        # FIRST VALID oversight removal: gate the enforcer bonus in ghost so
        # enforcement genuinely stops (all agents present = coord/distribution kept).
        ENV3C = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,
                              zap_removal_steps=25, bonus_requires_mark=True,
                              c_zapped=2.0, n_berry_types=3, grid=22,
                              ghost_keeps_bonus=False)
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        run_extinction([(1,), (0,)], ENV3C, hp, n_install=1000,
                       out_csv="extinct3_cleanghost.csv")
        analyze("extinct3_cleanghost.csv")
    elif a.enforceronly3:
        # enforcer-incentive-only removal: keep the violator cost (enforce stays ON,
        # n_install=updates so it never gates), withhold only the enforcer bonus at 1000.
        # The clean 2x2 confirmatory: if silly decays here, the norm rests on the
        # enforcer's incentive, not the violator's cost.
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        run_extinction([(1,), (0,)], ENV3, hp, n_install=1600, n_seeds=10,
                       gate_bonus_after=1000, out_csv="extinct3_enforceronly.csv")
        analyze("extinct3_enforceronly.csv")
    elif a.timeoutonly3:
        # gate ONLY the 25-step timeout removal; keep the zap penalty and the bonus.
        # Isolates whether "violator cost" includes the timeout.
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        run_extinction([(1,), (0,)], ENV3, hp, n_install=1600, n_seeds=10,
                       gate_removal_after=1000, out_csv="extinct3_timeoutonly.csv")
        analyze("extinct3_timeoutonly.csv")
    elif a.penaltyonly3:
        # gate ONLY the zap penalty (enforce off after 1000); keep timeout live and bonus.
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        run_extinction([(1,), (0,)], ENV3, hp, n_install=1000, n_seeds=10,
                       gate_removal_after=1600, out_csv="extinct3_penaltyonly.csv")
        analyze("extinct3_penaltyonly.csv")
    elif a.storage3:
        # storage test: full removal (bonus gated) + isolate (1 agent) + no-cue
        # (marks masked) + frozen weights, all at the switch. Fixed policy, alone, no
        # cue: does it still refuse berry-1 = stored-not-reconstructed.
        ENV3C = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,
                              zap_removal_steps=25, bonus_requires_mark=True,
                              c_zapped=2.0, n_berry_types=3, grid=22,
                              ghost_keeps_bonus=False)
        hp = dict(R.FAITHFUL_HP); hp["updates"] = 1600
        run_extinction([(1,), (0,)], ENV3C, hp, n_install=1000,
                       freeze_after=1000, isolate_after=1000, n_focal=1,
                       unmark_after=1000, out_csv="extinct3_storage.csv")
        analyze("extinct3_storage.csv")
    else:
        ap.error("pick --run3/--frozen3/--isolate3/--cleanghost3/--storage3/"
                 "--smoke3/--smokeiso3/--analyze CSV")
