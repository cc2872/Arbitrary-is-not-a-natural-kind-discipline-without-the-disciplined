"""Generate emergence_probe.ipynb -- the confirm/deny for EMERGENT selective
enforcement, as one self-contained Kaggle run-all notebook.

Question: with the zap-farm exploit turned DOWN (low r_zap_bonus) but at FAITHFUL
scale (updates=1500, num_envs=256, N=12, the Koster regime), does the population
develop SELECTIVE enforcement on its own -- selectivity > 1, agents preferentially
zapping MARKED agents -- and does that open a real per-encounter compliance RATE
gap? This is the non-scaffolded ("emergent") path. No code change: it just sweeps
r_zap_bonus in the low window at faithful scale and reads selectivity + rates.

CONFIRM = sel(0,1) robustly > 1 (mean - sd across 5 seeds) at some bonus.
DENY    = sel stays <= 1 everywhere -> emergence is dead at faithful scale;
          go scaffolded (contingent bonus) or bounded-null.

Runtime: 3 bonuses x 3 conditions x 5 seeds x 1.15e8 steps ~ 5e9 steps -- hours on
a P100, fits a 12 h Save & Run All. Shrink BONUS_LIST / UPDATES to go faster; each
bonus writes its own emerge_bonus_<b>.csv (bankable).

Regenerate:  python make_emergence_probe.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py"]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": list(lines)}


def writefile_cell(path):
    with open(path, encoding="utf-8") as f:
        body = f.read()
    return code(f"%%writefile {path}\n", *body.splitlines(keepends=True))


cells = [
    md("# Emergence probe - does selective enforcement arise on its own? (Kaggle)\n",
       "\n",
       "**Set GPU** (Session options -> Accelerator -> GPU), then **Save Version -> "
       "Save & Run All** for a detached 12 h batch (this is heavy).\n",
       "\n",
       "**The confirm/deny.** At `r_zap_bonus=8.75` agents zap-farm indiscriminately "
       "(selectivity ~0.95, no norm signal). This probe turns the bonus DOWN into "
       "the window where zapping is still worth it but farming isn't, runs at "
       "**faithful Koster scale** (updates=1500, num_envs=256, N=12, 5 seeds), and "
       "asks whether the population develops **selective** enforcement on its own.\n",
       "\n",
       "- **CONFIRM:** `sel(0,1)` robustly > 1 (mean-sd across seeds) at some bonus, "
       "and a per-encounter **rate** gap opens -> emergence is real, do the "
       "realistic hp search next.\n",
       "- **DENY:** selectivity stays <= 1 everywhere -> emergence is dead at "
       "faithful scale; go scaffolded (contingent bonus) or bounded-null.\n",
       "\n",
       "Self-contained (`%%writefile`); CSVs land in `/kaggle/working/`. Sequential "
       "seeds (memory-safe)."),

    md("## 0. Install (flax+optax; Kaggle's JAX already uses the GPU)"),
    code("!pip install -q flax optax\n",
         "import jax, numpy as np\n",
         "dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', (\n",
         "    'Still on CPU! Session options -> Accelerator -> GPU, then re-run.')\n",
         "print('GPU OK')"),

    md("## 1. Write the source files (self-contained; sequential seeds)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run (fast)"),
    code("import run_sweep as R\n",
         "hp = dict(R.FAITHFUL_HP); hp['num_envs'] = 16; hp['updates'] = 5\n",
         "_ = R.run_sweep([12], [(0,)], n_seeds=2, hp=hp,\n",
         "                env_list=[R.env_variant(poison_delay=100, r_zap_bonus=2.0,\n",
         "                          episode_len=300, zap_removal_steps=25)],\n",
         "                out_csv='sanity.csv')\n",
         "print('sanity OK')"),

    md("## 3. Emergence sweep (FAITHFUL scale - the heavy cell)\n",
       "Low-bonus window at faithful scale. Each bonus writes `emerge_bonus_<b>.csv`. "
       "Shrink `BONUS_LIST` or `UPDATES` if the session is tight."),
    code("import run_sweep as R\n",
         "\n",
         "BONUS_LIST = [0.5, 2, 4]      # low window (8.75 = known farming); shrink to fit\n",
         "UPDATES    = 1500            # faithful scale; lower to ~1000 to go faster\n",
         "\n",
         "for b in BONUS_LIST:\n",
         "    env = R.env_variant(poison_delay=100, r_zap_bonus=b,\n",
         "                        episode_len=300, zap_removal_steps=25)  # pending hidden\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = UPDATES; hp['num_envs'] = 256\n",
         "    print(f'\\n=== bonus={b}, faithful scale (N=12, 256 envs, {UPDATES} upd, 5 seeds) ===')\n",
         "    R.run_sweep([12], [(), (0,), (0, 1)], n_seeds=5, hp=hp,\n",
         "                env_list=[env], out_csv=f'emerge_bonus_{b}.csv')\n",
         "print('\\nDONE - emerge_bonus_*.csv written to /kaggle/working/')"),

    md("## 4. Verdict - did selective enforcement emerge, and is there a rate gap?\n",
       "`sel(0,1)` mean-sd > 1 = enforcement targets marked agents robustly. "
       "poison-RATE gap (none - 01) and berry-1 SHARE gap ((0,) - 01) > 0 with "
       "margin = a real per-encounter compliance effect (not foraging volume)."),
    code("import csv, glob, numpy as np\n",
         "from collections import defaultdict\n",
         "\n",
         "def cells(path, frac=0.1):\n",
         "    rows = list(csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(frac*(umax+1))\n",
         "    acc = defaultdict(lambda: defaultdict(lambda: [0.,0.,0.,0.,0]))  # e0,e1,sel,zap,n\n",
         "    for r in rows:\n",
         "        if int(r['update']) < cut: continue\n",
         "        a = acc[r['condition']][int(r['seed'])]\n",
         "        a[0]+=float(r['eat0']); a[1]+=float(r['eat1'])\n",
         "        a[2]+=float(r['selectivity']); a[3]+=float(r['zaps']); a[4]+=1\n",
         "    out = {}\n",
         "    for c in acc:\n",
         "        out[c] = {s:(v[0]/v[4], v[1]/v[4], v[2]/v[4], v[3]/v[4]) for s,v in acc[c].items()}\n",
         "    return out\n",
         "\n",
         "def paired(a, b):\n",
         "    seeds = sorted(set(a)&set(b)); d = np.array([a[s]-b[s] for s in seeds])\n",
         "    if len(d) < 2: return float(d.mean()), float('nan')\n",
         "    return float(d.mean()), float(d.std(ddof=1)/np.sqrt(len(d)))\n",
         "\n",
         "print(f\"{'bonus':>6} | {'sel(0,)':>14} {'sel(0,1)':>14} | {'zaps01':>7} | \"\n",
         "      f\"{'pois-rate gap':>13} {'b1-share gap':>13} | verdict\")\n",
         "print('-'*92)\n",
         "for path in sorted(glob.glob('emerge_bonus_*.csv'),\n",
         "                   key=lambda p: float(p.split('_')[-1][:-4])):\n",
         "    b = path.split('_')[-1][:-4]\n",
         "    C = cells(path)\n",
         "    def col(c, i): return {s:v[i] for s,v in C[c].items()}\n",
         "    s0 = np.array(list(col('0',2).values())); s01 = np.array(list(col('01',2).values()))\n",
         "    zap01 = np.mean(list(col('01',3).values()))\n",
         "    # rates per seed\n",
         "    pf = lambda c: {s:v[0]/(v[0]+v[1]) for s,v in C[c].items()}\n",
         "    b1 = lambda c: {s:v[1]/(v[0]+v[1]) for s,v in C[c].items()}\n",
         "    pr_m, pr_sem = paired(pf('none'), pf('01'))\n",
         "    b1_m, b1_sem = paired(b1('0'), b1('01'))\n",
         "    emerged = (s01.mean()-s01.std()) > 1.0\n",
         "    rate = (pr_m/pr_sem > 2) if pr_sem and np.isfinite(pr_sem) else False\n",
         "    v = 'EMERGES+RATE' if (emerged and rate) else ('sel>1 only' if emerged else 'flat')\n",
         "    print(f\"{b:>6} | {s0.mean():6.2f}+-{s0.std():4.2f} {s01.mean():6.2f}+-{s01.std():4.2f} | \"\n",
         "          f\"{zap01:7.1f} | {pr_m:+7.4f}/{pr_sem:.4f} {b1_m:+7.4f}/{b1_sem:.4f} | {v}\")\n",
         "print('\\nCONFIRM if any row is EMERGES+RATE (or at least sel>1 with the rate gap trending).')\n",
         "print('DENY if every row is flat -> emergence dead at faithful scale.')"),

    md("## 5. Outputs"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('emerge_bonus_*.csv')):\n",
         "    print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

with open("emergence_probe.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote emergence_probe.ipynb  (embedded:", ", ".join(SRC) + ")")
