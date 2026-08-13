"""Generate oppprobe2_gpu.ipynb -- the CLEAN silly-rule install confirmation.
Adds the (1,) silly-only condition so each rule's compliance is isolated with the
OTHER (unmarked) berry as its general-caution control -- removing the imperfect
poison-as-control caveat from the first opp run.

Conditions: none, (0,) poison-only, (1,) silly-only, (0,1) both. bonus 8.75
(the regime that installs selective enforcement). Verdict = clean diff-in-diff:

  POISON install (none->(0,)):  [opp0 drop] - [opp1 drop]   (berry-1 unmarked = caution)
  SILLY  install (none->(1,)):  [opp1 drop] - [opp0 drop]   (berry-0 unmarked = caution)

Both positive with margin>=2 = each rule installs a real per-encounter,
berry-specific norm. The SILLY one, airtight, is the flagship precondition.

Regenerate:  python make_oppprobe2_gpu.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py"]


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def writefile_cell(p):
    with open(p, encoding="utf-8") as f: body = f.read()
    return code(f"%%writefile {p}\n", *body.splitlines(keepends=True))


cells = [
    md("# Clean silly-rule confirmation (rented A100/H100)\n",
       "\n",
       "Adds the **`(1,)` silly-only** condition (mark berry-1, not poison) so the "
       "silly rule is tested with berry-0 as its own general-caution control — no "
       "reliance on poison-as-imperfect-control. bonus 8.75 (installs selective "
       "enforcement). Clean diff-in-diff:\n",
       "\n",
       "- **POISON** `none→(0,)`: `[opp0 drop] − [opp1 drop]` (berry-1 unmarked = caution)\n",
       "- **SILLY** `none→(1,)`: `[opp1 drop] − [opp0 drop]` (berry-0 unmarked = caution)\n",
       "\n",
       "Both positive, margin ≥ 2 = each installs a real berry-specific norm. The "
       "silly one airtight = flagship precondition. ~2.5 h (4 conditions)."),

    md("## 0. Install JAX-GPU + flax + optax"),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax; dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', 'No GPU - match jax[cudaXX] to image CUDA.'\n",
         "print('GPU OK:', dev[0].device_kind)"),

    md("## 1. Write the source files"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Run: bonus 8.75, 4 conditions (none, poison-only, silly-only, both)"),
    code("import run_sweep as R\n",
         "env = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
         "                    zap_removal_steps=25, bonus_requires_mark=True)\n",
         "hp = dict(R.FAITHFUL_HP); hp['updates'] = 1500; hp['num_envs'] = 256\n",
         "R.run_sweep([12], [(), (0,), (1,), (0, 1)], n_seeds=5, hp=hp,\n",
         "            env_list=[env], out_csv='opp2_8.75.csv', vmap_seeds=True)\n",
         "print('DONE - opp2_8.75.csv')"),

    md("## 3. Verdict - clean diff-in-diff (each rule vs its unmarked-berry control)"),
    code("import csv, numpy as np\n",
         "from collections import defaultdict\n",
         "rows = list(csv.DictReader(open('opp2_8.75.csv')))\n",
         "umax = max(int(r['update']) for r in rows); cut = umax - int(0.1*(umax+1))\n",
         "C = defaultdict(lambda: defaultdict(lambda: [0.,0.,0.,0.]))\n",
         "for r in rows:\n",
         "    if int(r['update']) < cut: continue\n",
         "    a = C[r['condition']][int(r['seed'])]\n",
         "    a[0]+=float(r['eat0']); a[1]+=float(r['eat1']); a[2]+=float(r['enc0']); a[3]+=float(r['enc1'])\n",
         "o0 = {c:{s:C[c][s][0]/max(C[c][s][2],1e-9) for s in C[c]} for c in C}\n",
         "o1 = {c:{s:C[c][s][1]/max(C[c][s][3],1e-9) for s in C[c]} for c in C}\n",
         "def M(v): return np.mean(v), (np.std(v,ddof=1)/np.sqrt(len(v)) if len(v)>1 else float('nan'))\n",
         "print(f\"{'cond':>5} | {'opp0(pois)':>10} {'opp1(silly)':>11}\")\n",
         "for c in ('none','0','1','01'):\n",
         "    if c in C: print(f\"{c:>5} | {np.mean(list(o0[c].values())):10.3f} {np.mean(list(o1[c].values())):11.3f}\")\n",
         "def dd(mark, ctrl, cond):  # marked-berry drop minus control-berry drop, none->cond\n",
         "    ss = sorted(set(o0['none']) & set(o0[cond]))\n",
         "    return M([(mark['none'][s]-mark[cond][s]) - (ctrl['none'][s]-ctrl[cond][s]) for s in ss])\n",
         "if '0' in C:\n",
         "    m,se = dd(o0, o1, '0'); print(f\"\\nPOISON install none->(0,) diff-in-diff: {m:+.4f} margin {m/se:+.2f}\")\n",
         "if '1' in C:\n",
         "    m,se = dd(o1, o0, '1'); v='SILLY RULE INSTALLS (airtight)' if m/se>=2 else 'silly flat'\n",
         "    print(f\"SILLY  install none->(1,) diff-in-diff: {m:+.4f} margin {m/se:+.2f}  <-- {v}\")"),

    md("## 4. Outputs"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('opp2_*.csv')): print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("oppprobe2_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote oppprobe2_gpu.ipynb")
