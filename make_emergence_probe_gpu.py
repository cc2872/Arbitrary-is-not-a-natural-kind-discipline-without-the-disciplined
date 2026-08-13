"""Generate emergence_probe_gpu.ipynb -- the emergence confirm/deny for a RENTED
cloud GPU (RunPod / Lambda / Vast: a bare Linux CUDA box). Differences vs the
Kaggle version:
  * JAX-GPU is NOT preinstalled -> pip install "jax[cuda12]" flax optax
  * 80 GB A100 -> vmap_seeds=True (drop the sequential 5x tax)
  * outputs are just local files in the working dir

Regenerate:  python make_emergence_probe_gpu.py
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
    md("# Emergence probe (rented A100) - does selective enforcement emerge?\n",
       "\n",
       "For a **rented cloud GPU** (RunPod / Lambda / Vast). Assumes an 80 GB A100 "
       "-> uses **vmapped seeds** (fast). If you're on a 40 GB card and it OOMs, set "
       "`VMAP_SEEDS = False` in section 3.\n",
       "\n",
       "**The confirm/deny.** Sweep `r_zap_bonus` in the low window (8.75 farms, 0 "
       "is dead) at **faithful scale** (updates=1500, num_envs=256, N=12, 5 seeds) "
       "and ask whether selective enforcement emerges on its own:\n",
       "- **CONFIRM:** `sel(0,1)` mean-sd > 1 at some bonus **and** a per-encounter "
       "rate gap opens.\n",
       "- **DENY:** flat everywhere -> emergence dead at faithful scale; go "
       "scaffolded or bounded-null.\n",
       "\n",
       "Self-contained (`%%writefile`). Each bonus writes `emerge_bonus_<b>.csv`."),

    md("## 0. Install JAX-GPU + flax + optax (rented box: nothing preinstalled)\n",
       "If your image already has a CUDA JAX, the reinstall is a no-op. For CUDA 11 "
       "images use `jax[cuda11_pip]` instead."),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax, numpy as np\n",
         "dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', (\n",
         "    'JAX is not seeing the GPU. Check the CUDA version of your image and\\n'\n",
         "    'match the jax[cudaXX] extra above (cuda12 vs cuda11).')\n",
         "print('GPU OK:', dev[0].device_kind)"),

    md("## 1. Write the source files (self-contained)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run"),
    code("import run_sweep as R\n",
         "hp = dict(R.FAITHFUL_HP); hp['num_envs'] = 16; hp['updates'] = 5\n",
         "_ = R.run_sweep([12], [(0,)], n_seeds=2, hp=hp,\n",
         "                env_list=[R.env_variant(poison_delay=100, r_zap_bonus=2.0,\n",
         "                          episode_len=300, zap_removal_steps=25)],\n",
         "                out_csv='sanity.csv', vmap_seeds=True)\n",
         "print('sanity OK')"),

    md("## 3. Emergence sweep (faithful scale, vmapped seeds)\n",
       "3 bonuses x 3 conditions x 5 seeds x 1.15e8 steps. On an 80 GB A100 with "
       "vmap this is ~1-3 h. Each bonus banks `emerge_bonus_<b>.csv`."),
    code("import run_sweep as R\n",
         "\n",
         "BONUS_LIST = [0.5, 2, 4]      # low window; shrink to fit\n",
         "UPDATES    = 1500            # faithful scale\n",
         "VMAP_SEEDS = True            # 80 GB A100 -> True (fast); 40 GB -> False\n",
         "\n",
         "for b in BONUS_LIST:\n",
         "    env = R.env_variant(poison_delay=100, r_zap_bonus=b,\n",
         "                        episode_len=300, zap_removal_steps=25)  # pending hidden\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = UPDATES; hp['num_envs'] = 256\n",
         "    print(f'\\n=== bonus={b}, faithful (N=12, 256 envs, {UPDATES} upd, 5 seeds, vmap={VMAP_SEEDS}) ===')\n",
         "    R.run_sweep([12], [(), (0,), (0, 1)], n_seeds=5, hp=hp,\n",
         "                env_list=[env], out_csv=f'emerge_bonus_{b}.csv',\n",
         "                vmap_seeds=VMAP_SEEDS)\n",
         "print('\\nDONE - emerge_bonus_*.csv written')"),

    md("## 4. Verdict - did selective enforcement emerge, and is there a rate gap?\n",
       "`sel(0,1)` mean-sd > 1 = enforcement robustly targets marked agents. "
       "poison-RATE gap (none-01) and berry-1 SHARE gap ((0,)-01) with margin>=2 = "
       "a real per-encounter compliance effect (not foraging volume)."),
    code("import csv, glob, numpy as np\n",
         "from collections import defaultdict\n",
         "\n",
         "def cells(path, frac=0.1):\n",
         "    rows = list(csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(frac*(umax+1))\n",
         "    acc = defaultdict(lambda: defaultdict(lambda: [0.,0.,0.,0.,0]))\n",
         "    for r in rows:\n",
         "        if int(r['update']) < cut: continue\n",
         "        a = acc[r['condition']][int(r['seed'])]\n",
         "        a[0]+=float(r['eat0']); a[1]+=float(r['eat1'])\n",
         "        a[2]+=float(r['selectivity']); a[3]+=float(r['zaps']); a[4]+=1\n",
         "    return {c:{s:(v[0]/v[4], v[1]/v[4], v[2]/v[4], v[3]/v[4]) for s,v in acc[c].items()} for c in acc}\n",
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
         "    b = path.split('_')[-1][:-4]; C = cells(path)\n",
         "    def col(c, i): return {s:v[i] for s,v in C[c].items()}\n",
         "    s0 = np.array(list(col('0',2).values())); s01 = np.array(list(col('01',2).values()))\n",
         "    zap01 = np.mean(list(col('01',3).values()))\n",
         "    pf = lambda c: {s:v[0]/(v[0]+v[1]) for s,v in C[c].items()}\n",
         "    b1 = lambda c: {s:v[1]/(v[0]+v[1]) for s,v in C[c].items()}\n",
         "    pr_m, pr_sem = paired(pf('none'), pf('01'))\n",
         "    b1_m, b1_sem = paired(b1('0'), b1('01'))\n",
         "    emerged = (s01.mean()-s01.std()) > 1.0\n",
         "    rate = (abs(pr_m)/pr_sem > 2) if pr_sem and np.isfinite(pr_sem) else False\n",
         "    v = 'EMERGES+RATE' if (emerged and rate) else ('sel>1 only' if emerged else 'flat')\n",
         "    print(f\"{b:>6} | {s0.mean():6.2f}+-{s0.std():4.2f} {s01.mean():6.2f}+-{s01.std():4.2f} | \"\n",
         "          f\"{zap01:7.1f} | {pr_m:+7.4f}/{pr_sem:.4f} {b1_m:+7.4f}/{b1_sem:.4f} | {v}\")\n",
         "print('\\nCONFIRM if any row is EMERGES+RATE. DENY if all flat -> emergence dead at faithful scale.')"),

    md("## 5. Outputs (download from the pod, or push to your Drive/repo)"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('emerge_bonus_*.csv')):\n",
         "    print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

with open("emergence_probe_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote emergence_probe_gpu.ipynb  (embedded:", ", ".join(SRC) + ")")
