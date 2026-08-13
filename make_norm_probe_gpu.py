"""Generate norm_probe_gpu.ipynb -- STEP 2: does MARK-CONTINGENT (faithful)
enforcement install a norm? For a rented A100. Same substrate as the emergence
probe, but with the fidelity fix ON: bonus_requires_mark=True pays r_zap_bonus
only for zapping a MARKED target (Koster's incentive), instead of the flat bonus
that flattened selectivity at every level.

CONFIRM (a norm installs) = sel(0,) mean-sd > 1 (agents now target violators;
sel(0,) is the honest, less base-rate-compressed metric) AND a per-encounter RATE
gap opens (poison_frac none-01 or berry-1 share (0,)-01, margin>=2). Then the
ghost cell finally has something to remove.

Regenerate:  python make_norm_probe_gpu.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py"]


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def writefile_cell(p):
    with open(p, encoding="utf-8") as f: body = f.read()
    return code(f"%%writefile {p}\n", *body.splitlines(keepends=True))


cells = [
    md("# Norm-install probe (rented A100) - does mark-contingent enforcement install a norm?\n",
       "\n",
       "STEP 2 of: fix reward -> **confirm a norm installs** -> ghost cell -> extinction.\n",
       "\n",
       "The flat-bonus env never produced selective enforcement (sel(0,) capped "
       "under 1 at every bonus) because the reward paid for *any* landed zap. This "
       "runs the fidelity fix: `bonus_requires_mark=True` pays `r_zap_bonus` only "
       "for zapping a **marked** target (Koster's +marked / -unmarked incentive), "
       "at faithful scale (updates=1500, num_envs=256, N=12, 5 seeds).\n",
       "\n",
       "- **CONFIRM (norm installs):** `sel(0,)` mean-sd > 1 (agents target "
       "violators) **and** a per-encounter RATE gap opens.\n",
       "- **still flat:** even the correct incentive doesn't install at this scale "
       "-> scale up, or the reduced env can't support it (a bounded result).\n",
       "\n",
       "Read `sel(0,)`, not `sel(0,1)` - the latter compresses toward 1 by base "
       "rate (both berries marked). 80 GB A100 -> vmapped seeds."),

    md("## 0. Install JAX-GPU + flax + optax"),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax, numpy as np\n",
         "dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', 'No GPU - match jax[cudaXX] to the image CUDA.'\n",
         "print('GPU OK:', dev[0].device_kind)"),

    md("## 1. Write the source files (with the mark-contingent fix)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run (mark-contingent path)"),
    code("import run_sweep as R\n",
         "hp = dict(R.FAITHFUL_HP); hp['num_envs'] = 16; hp['updates'] = 5\n",
         "_ = R.run_sweep([12], [(0,)], n_seeds=2, hp=hp,\n",
         "                env_list=[R.env_variant(poison_delay=100, r_zap_bonus=8.75,\n",
         "                          episode_len=300, zap_removal_steps=25,\n",
         "                          bonus_requires_mark=True)],\n",
         "                out_csv='sanity.csv', vmap_seeds=True)\n",
         "print('sanity OK (mark-contingent)')"),

    md("## 3. Norm-install sweep (faithful scale, MARK-CONTINGENT bonus)\n",
       "`bonus_requires_mark=True`. Bonuses bracket low/high; no farming risk now "
       "(unmarked zaps pay nothing). Each writes `mark_bonus_<b>.csv`."),
    code("import run_sweep as R\n",
         "\n",
         "BONUS_LIST = [2, 8.75]       # marked-only payoff; add 0.5/4 if you want the curve\n",
         "UPDATES    = 1500\n",
         "\n",
         "for b in BONUS_LIST:\n",
         "    env = R.env_variant(poison_delay=100, r_zap_bonus=b, episode_len=300,\n",
         "                        zap_removal_steps=25, bonus_requires_mark=True)\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = UPDATES; hp['num_envs'] = 256\n",
         "    print(f'\\n=== bonus={b} (MARK-CONTINGENT), faithful, 5 seeds ===')\n",
         "    R.run_sweep([12], [(), (0,), (0, 1)], n_seeds=5, hp=hp,\n",
         "                env_list=[env], out_csv=f'mark_bonus_{b}.csv', vmap_seeds=True)\n",
         "print('\\nDONE - mark_bonus_*.csv written')"),

    md("## 4. Verdict - did a norm install? (sel read against its achievable ceiling)\n",
       "`sel(0,)` = share/prev; **random targeting = 1.00, perfect targeting = "
       "ceil = 1/prev.** So `sel` is read against `ceil`, not a bare 1 - if `ceil` "
       "is ~1.1 the metric is prevalence-compressed (a small `sel` gain still means "
       "strong targeting); if `ceil` > ~1.5 there's real room, so `sel<=1` is "
       "genuinely no targeting. NORM INSTALLS = `sel(0,)` mean-sd > 1 AND a "
       "per-encounter RATE gap (poison none-01 or berry-1 (0,)-01, margin>=2)."),
    code("import csv, glob, numpy as np\n",
         "from collections import defaultdict\n",
         "\n",
         "def cells(path, frac=0.1):\n",
         "    rows = list(csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(frac*(umax+1))\n",
         "    acc = defaultdict(lambda: defaultdict(lambda: [0.,0.,0.,0.,0.,0]))\n",
         "    for r in rows:\n",
         "        if int(r['update']) < cut: continue\n",
         "        a = acc[r['condition']][int(r['seed'])]\n",
         "        a[0]+=float(r['eat0']); a[1]+=float(r['eat1']); a[2]+=float(r['selectivity'])\n",
         "        a[3]+=float(r['zaps']); a[4]+=float(r.get('prevalence',0) or 0); a[5]+=1\n",
         "    return {c:{s:(v[0]/v[5],v[1]/v[5],v[2]/v[5],v[3]/v[5],v[4]/v[5]) for s,v in acc[c].items()} for c in acc}\n",
         "\n",
         "def paired(a, b):\n",
         "    s = sorted(set(a)&set(b)); d = np.array([a[x]-b[x] for x in s])\n",
         "    return (float(d.mean()), float(d.std(ddof=1)/np.sqrt(len(d)))) if len(d)>1 else (float(d.mean()), float('nan'))\n",
         "\n",
         "print(f\"{'bonus':>6} | {'sel(0,)':>12} {'prev':>5} {'ceil':>5} {'%room':>6} | {'zaps0':>6} | \"\n",
         "      f\"{'pois-rate gap':>13} {'b1-share gap':>13} | verdict\")\n",
         "print('-'*104)\n",
         "for path in sorted(glob.glob('mark_bonus_*.csv'), key=lambda p: float(p.split('_')[-1][:-4])):\n",
         "    b = path.split('_')[-1][:-4]; C = cells(path)\n",
         "    def col(c,i): return {s:v[i] for s,v in C[c].items()}\n",
         "    s0 = np.array(list(col('0',2).values())); p0 = float(np.mean(list(col('0',4).values())))\n",
         "    z0 = float(np.mean(list(col('0',3).values()))); ceil = 1.0/max(p0,1e-6)\n",
         "    room = (s0.mean()-1)/(ceil-1)*100 if ceil>1.001 else float('nan')  # %% of room used above random\n",
         "    pf = lambda c: {s:v[0]/(v[0]+v[1]) for s,v in C[c].items()}\n",
         "    b1 = lambda c: {s:v[1]/(v[0]+v[1]) for s,v in C[c].items()}\n",
         "    prm, prs = paired(pf('none'), pf('01')); b1m, b1s = paired(b1('0'), b1('01'))\n",
         "    installed = (s0.mean()-s0.std()) > 1.0\n",
         "    rate = (abs(prm)/prs > 2) if prs and np.isfinite(prs) else False\n",
         "    v = 'NORM INSTALLS' if (installed and rate) else ('sel>1 only' if installed else 'flat')\n",
         "    print(f\"{b:>6} | {s0.mean():5.2f}+-{s0.std():4.2f} {p0:5.2f} {ceil:5.2f} {room:6.0f} | \"\n",
         "          f\"{z0:6.1f} | {prm:+7.4f}/{prs:.4f} {b1m:+7.4f}/{b1s:.4f} | {v}\")\n",
         "print('\\n%room = how far sel sits from random(1) toward perfect(ceil). If ceil~1.1 the metric is')\n",
         "print('prevalence-compressed; if ceil>1.5 sel<=1 is genuinely no targeting. NORM INSTALLS = the goal.')"),

    md("## 5. Outputs"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('mark_bonus_*.csv')):\n",
         "    print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("norm_probe_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote norm_probe_gpu.ipynb  (embedded:", ", ".join(SRC) + ")")
