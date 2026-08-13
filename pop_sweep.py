"""
pop_sweep.py -- Phase-2 pilot #1: does the silly-rule norm survive at smaller N?

Runs at the EXACT conditions that produced koster_replication.csv (the trustworthy
dissociation: none eat0 ~91, (0,1) ~45). Forensic audit (2026-08-01) pinned those
conditions from the CSV columns + git:

    run_sweep([12], [(), (0,), (0, 1)], n_seeds=5,
              hp = FAITHFUL_HP with updates=800,
              env = env_variant(poison_delay=100, r_zap_bonus=8.75,
                                episode_len=300, zap_removal_steps=25))
    code = berryworld_jax@a954774 + train_jax@d2b0ee1  (== current, bit-exact)

Every git-tracked knob (env code, train_jax, all named FAITHFUL_HP values) is
IDENTICAL between koster (none->91) and confirm256 (none->43). observe_pending is
RULED OUT (a954774 already zeroed pending -> koster ran pending-HIDDEN, same as
confirm256). The ONE degree of freedom the CSV never recorded is num_envs -- and
run_sweep.py's own note says "num_envs was the difference that flipped the whole
result." So num_envs is a first-class parameter here; the koster config is
recovered by finding the num_envs that reproduces none->91 (see gate_2.ipynb 3).
If no num_envs recovers it, the 91 is a session/GPU bistability artifact, not a
config -- and koster cannot anchor the replication.

The statistic is the Gate-1 silly-rule statistic read PAIRED: run_sweep splits the
same PRNGKey(0) per cell, so seed s is the same world in (0,) and (0,1), and the
berry-1-specific norm signal is the within-seed difference

    d1_s = eat1_s[(0,)] - eat1_s[(0,1)]        # >0  <=> marking berry 1 suppresses it

averaged over seeds. Pairing cancels between-seed variance, so a real effect shows
at n=5.

Run (GPU):     python pop_sweep.py --run --num_envs 256   # writes pop_sweep_ne256.csv
Analyze:       python pop_sweep.py --analyze pop_sweep_ne256.csv
Local sanity:  python pop_sweep.py --smoke                # tiny, CPU-allowed
"""
import argparse
import csv
import numpy as np
import run_sweep as R

# N grid for the pilot: 6 is the extinction-experiment spec (the small-N risk,
# since the silly-rule effect grows with population); 8 and 10 bracket up to the
# N=12 replication anchor.
N_GRID = [6, 8, 10, 12]
CONDITIONS = [(), (0,), (0, 1)]
N_SEEDS = 5

# The exact koster_replication env: pending HIDDEN (observe_pending defaults False,
# git-confirmed as how a954774 ran), D=100, bonus=8.75, ep=300, removal=25.
POP_ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75,
                        episode_len=300, zap_removal_steps=25)


def make_hp(num_envs=256, updates=800):
    """FAITHFUL_HP at the koster training budget; num_envs is the recovered knob."""
    hp = dict(R.FAITHFUL_HP)
    hp["updates"] = updates
    hp["num_envs"] = num_envs
    return hp


def run(n_list=N_GRID, n_seeds=N_SEEDS, num_envs=256, updates=800, out_csv=None):
    if out_csv is None:
        out_csv = f"pop_sweep_ne{num_envs}.csv"      # self-documenting per num_envs
    return R.run_sweep(n_list, CONDITIONS, n_seeds=n_seeds,
                       hp=make_hp(num_envs, updates), env_list=[POP_ENV],
                       out_csv=out_csv)


# ---------------------------------------------------------------- analysis
def _tail_cells(csv_path, tail_frac=0.1):
    """Per (N, condition, seed) tail-mean of eat0/eat1 over the last tail_frac
    of updates. Returns nested dict cells[N][cond][seed] = (eat0, eat1)."""
    rows = list(csv.DictReader(open(csv_path)))
    umax = max(int(r["update"]) for r in rows)
    cut = umax - int(tail_frac * (umax + 1))
    acc = {}
    for r in rows:
        if int(r["update"]) < cut:
            continue
        k = (int(r["N"]), r["condition"], int(r["seed"]))
        acc.setdefault(k, []).append((float(r["eat0"]), float(r["eat1"])))
    cells = {}
    for (N, cond, s), vals in acc.items():
        a = np.array(vals)
        cells.setdefault(N, {}).setdefault(cond, {})[s] = a.mean(0)  # (eat0,eat1)
    return cells


def _paired(a_by_seed, b_by_seed):
    """Paired (a - b) over shared seeds: (mean, sem, n)."""
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    d = np.array([a_by_seed[s] - b_by_seed[s] for s in seeds])
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def analyze(csv_path):
    cells = _tail_cells(csv_path)
    print(f"\n=== pilot #1: silly-rule norm vs population size ({csv_path}) ===")
    print("berry-1-SPECIFIC norm signal = eat1[(0,)] - eat1[(0,1)], PAIRED over "
          "seeds.\n>0 means marking berry 1 suppresses eating it = a social norm "
          "exists to extinguish.\nverdict SURVIVES if margin (mean/SEM) >= 2 and "
          "sign positive.\n")
    print(f"{'N':>3} | {'eat1 (0,)':>10} {'eat1 (0,1)':>11} | "
          f"{'paired d1':>10} {'SEM':>6} {'margin':>7} | {'verdict':>9}")
    print("-" * 74)
    for N in sorted(cells):
        c = cells[N]
        if "0" not in c or "01" not in c:
            print(f"{N:>3} | incomplete (need both (0,) and (0,1) cells)")
            continue
        e1_0 = {s: v[1] for s, v in c["0"].items()}     # eat1 under (0,)
        e1_01 = {s: v[1] for s, v in c["01"].items()}   # eat1 under (0,1)
        m, sem, n = _paired(e1_0, e1_01)
        margin = m / sem if sem and np.isfinite(sem) and sem > 0 else float("nan")
        surv = "SURVIVES" if (np.isfinite(margin) and margin >= 2) else "weak/none"
        mean0 = np.mean([v for v in e1_0.values()])
        mean01 = np.mean([v for v in e1_01.values()])
        print(f"{N:>3} | {mean0:>10.1f} {mean01:>11.1f} | "
              f"{m:>10.2f} {sem:>6.2f} {margin:>7.2f} | {surv:>9}")
    print("\n(eat0/poison compliance across conditions -- the regime check:)")
    print("koster regime = none eat0 ~90 (fails unaided) >> (0,1) ~45.")
    print(f"{'N':>3} | {'none':>7} {'(0,)':>7} {'(0,1)':>7}")
    for N in sorted(cells):
        c = cells[N]
        def m0(cond):
            return np.mean([v[0] for v in c[cond].values()]) if cond in c else float("nan")
        print(f"{N:>3} | {m0('none'):>7.1f} {m0('0'):>7.1f} {m0('01'):>7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="GPU: full N-sweep at the koster config")
    ap.add_argument("--num_envs", type=int, default=256,
                    help="num_envs (the recovered knob); default 256")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-allowed sanity run")
    ap.add_argument("--analyze", metavar="CSV", help="read a finished CSV and report")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze)
    elif a.smoke:
        run(n_list=[6], n_seeds=2, num_envs=8, updates=20, out_csv="pop_sweep_smoke.csv")
        analyze("pop_sweep_smoke.csv")
    elif a.run:
        run(num_envs=a.num_envs)
        analyze(f"pop_sweep_ne{a.num_envs}.csv")
    else:
        ap.error("pick one of --run / --smoke / --analyze CSV")
