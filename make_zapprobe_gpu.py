"""Generate zapprobe_gpu.ipynb -- does a STRONGER zap penalty install the silly rule?
The silly rule fails because the harmless berry's only avoidance signal is the zap,
and the eat->mark->zap credit chain is too weak to install a berry-specific taboo
without an intrinsic anchor (poison has one). c_zapped (target penalty) is 2.0 here
vs Koster's -35 -- a huge gap. This sweeps c_zapped in {2, 15, 35} at the airtight
silly test (none, (0,), (1,); diff-in-diff with the unmarked berry as control).

SILLY install = opp1 drop none->(1,) MINUS opp0 (unmarked) drop, margin>=2.
Question: does the silly diff-in-diff trend off zero as the zap bites harder?

Same compute per run as any faithful run (~1 h each on H100), so all three are a
few hours total -- far cheaper than scaling steps. bonus 8.75 mark-contingent.

Regenerate:  python make_zapprobe_gpu.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py"]


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def writefile_cell(p):
    with open(p, encoding="utf-8") as f: body = f.read()
    return code(f"%%writefile {p}\n", *body.splitlines(keepends=True))


cells = [
    md("# Zap-strength probe: does a harder punishment install the silly rule?\n",
       "\n",
       "The silly rule (harmless berry-1) doesn't install: at `c_zapped=2` the zap "
       "alone can't carve a berry-specific taboo (poison installs because it *also* "
       "has an intrinsic penalty). This sweeps the target penalty **`c_zapped ∈ "
       "{2, 15, 35}`** (Köster's was −35) at the airtight silly test — conditions "
       "`none, (0,), (1,)`, diff-in-diff with the *unmarked* berry as the "
       "general-caution control.\n",
       "\n",
       "**SILLY installs** if `none→(1,)` diff-in-diff (opp1 drop − opp0 drop) "
       "clears margin ≥ 2. Watch whether it **trends off zero** as the zap bites "
       "harder. ~1 h/run on H100. Same-magnitude caveat: 35 is Köster-faithful, not "
       "engineering — but report the whole curve, not just the best point."),

    md("## 0. Install JAX-GPU + flax + optax"),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax; dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', 'No GPU - match jax[cudaXX] to image CUDA.'\n",
         "print('GPU OK:', dev[0].device_kind)"),

    md("## 1. Write the source files (c_zapped now threaded through env_variant)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sweep c_zapped (in-kernel, so no preallocation clash)"),
    code("import run_sweep as R\n",
         "CZAP = [2, 15, 35]        # target zap penalty; Koster = 35\n",
         "for cz in CZAP:\n",
         "    env = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
         "                        zap_removal_steps=25, bonus_requires_mark=True, c_zapped=float(cz))\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = 1500; hp['num_envs'] = 256\n",
         "    print(f'\\n=== c_zapped={cz} (bonus 8.75 mark-contingent), 5 seeds ===')\n",
         "    R.run_sweep([12], [(), (0,), (1,)], n_seeds=5, hp=hp,\n",
         "                env_list=[env], out_csv=f'zap_c{cz}.csv', vmap_seeds=True)\n",
         "print('\\nDONE - zap_c*.csv written')"),

    md("## 3. Verdict - silly (and poison) install vs zap strength"),
    code("import csv, glob, numpy as np\n",
         "from collections import defaultdict\n",
         "def load(path):\n",
         "    rows = list(csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(0.1*(umax+1))\n",
         "    C = defaultdict(lambda: defaultdict(lambda: [0.,0.,0.,0.]))\n",
         "    for r in rows:\n",
         "        if int(r['update']) < cut: continue\n",
         "        a = C[r['condition']][int(r['seed'])]\n",
         "        a[0]+=float(r['eat0']); a[1]+=float(r['eat1']); a[2]+=float(r['enc0']); a[3]+=float(r['enc1'])\n",
         "    o0 = {c:{s:C[c][s][0]/max(C[c][s][2],1e-9) for s in C[c]} for c in C}\n",
         "    o1 = {c:{s:C[c][s][1]/max(C[c][s][3],1e-9) for s in C[c]} for c in C}\n",
         "    return o0, o1\n",
         "def M(v): return np.mean(v), (np.std(v,ddof=1)/np.sqrt(len(v)) if len(v)>1 else float('nan'))\n",
         "def dd(mark, ctrl, cond):\n",
         "    ss = sorted(set(mark['none']) & set(mark[cond]))\n",
         "    return M([(mark['none'][s]-mark[cond][s]) - (ctrl['none'][s]-ctrl[cond][s]) for s in ss])\n",
         "print(f\"{'c_zap':>6} | {'opp1[none]':>10} {'opp1[(1,)]':>10} | {'POISON n->(0,)':>16} | {'SILLY n->(1,)':>16} | verdict\")\n",
         "for path in sorted(glob.glob('zap_c*.csv'), key=lambda p: float(p.split('_c')[1][:-4])):\n",
         "    cz = path.split('_c')[1][:-4]; o0,o1 = load(path)\n",
         "    if '1' not in o1: print(f\"{cz:>6} | (incomplete)\"); continue\n",
         "    pm,ps = dd(o0,o1,'0') if '0' in o0 else (float('nan'),float('nan'))\n",
         "    sm,ss = dd(o1,o0,'1')\n",
         "    v = 'SILLY INSTALLS' if (np.isfinite(ss) and sm/ss>=2) else 'silly flat'\n",
         "    on=np.mean(list(o1['none'].values())); o1c=np.mean(list(o1['1'].values()))\n",
         "    print(f\"{cz:>6} | {on:10.3f} {o1c:10.3f} | {pm:+7.4f} m{pm/ps:+5.1f} | {sm:+7.4f} m{sm/ss:+5.1f} | {v}\")\n",
         "print('\\ntrend: does SILLY diff-in-diff climb off zero as c_zapped rises? monotone up = punishment strength is the lever.')"),

    md("## 4. Outputs"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('zap_c*.csv')): print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("zapprobe_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote zapprobe_gpu.ipynb")
