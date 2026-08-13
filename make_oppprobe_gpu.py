"""Generate oppprobe_gpu.ipynb -- the flagship precondition test on a rented A100.
Re-runs the mark-contingent enforcement config that installed a poison norm
(bonus 8.75), now with the OPPORTUNITY-CONTROLLED metric, to ask the question the
diet-share DV couldn't: does the PURELY-SOCIAL silly rule (harmless berry-1)
install as a per-encounter compliance effect?

DV = eat_t / encounter_t = P(eat berry-t | standing on a berry-t cell), decoupled
from diet share (which is confounded by 2-berry complementarity). New CSV cols
enc0/enc1 (encounters per type) make this computable.

  POISON opp compliance  = opp0[none] - opp0[(0,)]   (marking poison lowers its eat-rate)
  SILLY  opp compliance  = opp1[(0,)] - opp1[(0,1)]  (marking harmless berry-1 lowers ITS eat-rate)  <-- flagship DV

If SILLY clears margin>=2 positive, a purely-social norm installs -> a clean
constraint to extinguish. If poison installs but silly doesn't, that is itself a
sharp result about which norms enforcement can install.

Regenerate:  python make_oppprobe_gpu.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py"]


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def writefile_cell(p):
    with open(p, encoding="utf-8") as f: body = f.read()
    return code(f"%%writefile {p}\n", *body.splitlines(keepends=True))


cells = [
    md("# Opportunity-controlled probe (rented A100): does the SILLY rule install?\n",
       "\n",
       "Mark-contingent enforcement at bonus=8.75 installed a real poison norm "
       "(`sel 1.59`, poison rate 0.359→0.189, margin +18.6). This re-runs it with "
       "the **opportunity-controlled** metric — `eat_t / encounter_t` = P(eat "
       "berry-t | standing on it) — to test the flagship precondition: does the "
       "**purely-social** silly rule (harmless berry-1) reduce its *per-encounter* "
       "eat rate when marked? Diet-share couldn't answer this (berry-1 share ≡ "
       "1 − poison_frac).\n",
       "\n",
       "**SILLY opp compliance = opp1[(0,)] − opp1[(0,1)]**, margin ≥ 2 positive = "
       "a real social norm installs → a clean constraint to extinguish. 80 GB A100 "
       "→ vmapped seeds; ~2 h."),

    md("## 0. Install JAX-GPU + flax + optax"),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax, numpy as np\n",
         "dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', 'No GPU - match jax[cudaXX] to the image CUDA.'\n",
         "print('GPU OK:', dev[0].device_kind)"),

    md("## 1. Write the source files (with enc0/enc1 opportunity logging)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run (confirms enc columns present)"),
    code("import run_sweep as R\n",
         "hp = dict(R.FAITHFUL_HP); hp['num_envs'] = 16; hp['updates'] = 5\n",
         "rows = R.run_sweep([12], [(0, 1)], n_seeds=2, hp=hp,\n",
         "                   env_list=[R.env_variant(poison_delay=100, r_zap_bonus=8.75,\n",
         "                             episode_len=300, zap_removal_steps=25,\n",
         "                             bonus_requires_mark=True)],\n",
         "                   out_csv='sanity.csv', vmap_seeds=True)\n",
         "assert 'enc1' in rows[0], 'enc columns missing'\n",
         "print('sanity OK; cols:', [c for c in rows[0] if c in ('eat1','enc1','selectivity')])"),

    md("## 3. Run: mark-contingent, opportunity-logged (faithful scale)\n",
       "bonus ∈ {2, 8.75} (dose-response), `bonus_requires_mark=True`, 3 conditions "
       "× 5 seeds. Each writes `opp_bonus_<b>.csv`."),
    code("import run_sweep as R\n",
         "\n",
         "BONUS_LIST = [2, 8.75]\n",
         "for b in BONUS_LIST:\n",
         "    env = R.env_variant(poison_delay=100, r_zap_bonus=b, episode_len=300,\n",
         "                        zap_removal_steps=25, bonus_requires_mark=True)\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = 1500; hp['num_envs'] = 256\n",
         "    print(f'\\n=== bonus={b} (mark-contingent, opportunity-logged), 5 seeds ===')\n",
         "    R.run_sweep([12], [(), (0,), (0, 1)], n_seeds=5, hp=hp,\n",
         "                env_list=[env], out_csv=f'opp_bonus_{b}.csv', vmap_seeds=True)\n",
         "print('\\nDONE - opp_bonus_*.csv written')"),

    md("## 4. Verdict - opportunity-controlled compliance (poison vs silly)\n",
       "`opp_t = sum(eat_t)/sum(enc_t)` per seed over the tail; paired across seeds. "
       "SILLY DV is the flagship precondition."),
    code("import csv, glob, numpy as np\n",
         "from collections import defaultdict\n",
         "\n",
         "def tail(path, frac=0.1):\n",
         "    rows = list(csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(frac*(umax+1))\n",
         "    acc = defaultdict(lambda: defaultdict(lambda: [0.,0.,0.,0.,0.,0]))  # e0,e1,c0,c1,sel,n\n",
         "    for r in rows:\n",
         "        if int(r['update']) < cut: continue\n",
         "        a = acc[r['condition']][int(r['seed'])]\n",
         "        a[0]+=float(r['eat0']); a[1]+=float(r['eat1']); a[2]+=float(r['enc0']); a[3]+=float(r['enc1'])\n",
         "        a[4]+=float(r['selectivity']); a[5]+=1\n",
         "    return acc\n",
         "\n",
         "def paired(a, b):\n",
         "    s = sorted(set(a)&set(b)); d = np.array([a[x]-b[x] for x in s])\n",
         "    return d.mean(), (d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan'))\n",
         "\n",
         "for path in sorted(glob.glob('opp_bonus_*.csv'), key=lambda p: float(p.split('_')[-1][:-4])):\n",
         "    b = path.split('_')[-1][:-4]; acc = tail(path)\n",
         "    opp0 = {c:{s:acc[c][s][0]/max(acc[c][s][2],1e-9) for s in acc[c]} for c in acc}\n",
         "    opp1 = {c:{s:acc[c][s][1]/max(acc[c][s][3],1e-9) for s in acc[c]} for c in acc}\n",
         "    sel0 = np.mean([acc['0'][s][4]/acc['0'][s][5] for s in acc['0']]) if '0' in acc else float('nan')\n",
         "    print(f\"\\n=== bonus={b}   sel(0,)={sel0:.2f} ===\")\n",
         "    print(f\"{'cond':>5} | {'opp0 (poison eat|enc)':>22} {'opp1 (silly eat|enc)':>22}\")\n",
         "    for c in ('none','0','01'):\n",
         "        if c not in acc: continue\n",
         "        o0 = np.mean(list(opp0[c].values())); o1 = np.mean(list(opp1[c].values()))\n",
         "        print(f\"{c:>5} | {o0:22.3f} {o1:22.3f}\")\n",
         "    if {'none','0'} <= set(acc):\n",
         "        pm, ps = paired(opp0['none'], opp0['0'])\n",
         "        print(f\"  POISON opp compliance none-(0,): {pm:+.4f} margin {pm/ps:+.2f}\")\n",
         "    if {'0','01'} <= set(acc):\n",
         "        sm, ss = paired(opp1['0'], opp1['01'])\n",
         "        verdict = 'SILLY RULE INSTALLS' if (np.isfinite(ss) and sm/ss >= 2) else 'silly flat/absent'\n",
         "        print(f\"  SILLY  opp compliance (0,)-(01): {sm:+.4f} margin {sm/ss:+.2f}  <-- {verdict}\")"),

    md("## 5. Outputs"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('opp_bonus_*.csv')):\n",
         "    print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("oppprobe_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote oppprobe_gpu.ipynb  (embedded:", ", ".join(SRC) + ")")
