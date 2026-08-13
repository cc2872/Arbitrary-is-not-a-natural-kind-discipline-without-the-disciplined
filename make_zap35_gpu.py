"""Generate zap35_gpu.ipynb -- one-off: does c_zapped=35 (Koster's target penalty)
install the silly rule? Airtight test only, single config, ~1 h on H100. Run on a
separate pod in parallel with the full sweep for the fast answer.
"""
import json
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py"]
def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def wf(p):
    with open(p, encoding="utf-8") as f: b=f.read()
    return code(f"%%writefile {p}\n", *b.splitlines(keepends=True))

cells = [
    md("# One-off: does c_zapped=35 install the silly rule?\n",
       "\n",
       "The mechanistically-direct lever. Target zap penalty **35** (Köster's −35, "
       "vs the default 2) at the airtight silly test — `none, (0,), (1,)`, "
       "diff-in-diff with the *unmarked* berry as caution control, bonus 8.75 "
       "mark-contingent, faithful scale, 5 seeds. ~1 h, in-kernel (no OOM).\n",
       "\n",
       "**SILLY INSTALLS** if `none→(1,)` diff-in-diff (opp1 drop − opp0 drop) "
       "clears margin ≥ 2. Baseline at c=2 is ~0."),
    md("## 0. Environment"),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax; dev=jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform=='gpu', 'No GPU'\n",
         "print('GPU OK:', dev[0].device_kind)"),
    md("## 1. Sources"),
    *[wf(p) for p in SRC],
    md("## 2. Run c_zapped=35 (in-kernel)"),
    code("import run_sweep as R\n",
         "env = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
         "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=35.0)\n",
         "hp = dict(R.FAITHFUL_HP); hp['updates']=1500; hp['num_envs']=256\n",
         "R.run_sweep([12], [(), (0,), (1,)], n_seeds=5, hp=hp,\n",
         "            env_list=[env], out_csv='zap_c35.csv', vmap_seeds=True)\n",
         "print('DONE - zap_c35.csv')"),
    md("## 3. Verdict"),
    code("import csv, numpy as np\n",
         "from collections import defaultdict\n",
         "rows=list(csv.DictReader(open('zap_c35.csv')))\n",
         "umax=max(int(r['update']) for r in rows); cut=umax-int(0.1*(umax+1))\n",
         "C=defaultdict(lambda: defaultdict(lambda:[0.,0.,0.,0.,0.,0.]))\n",
         "for r in rows:\n",
         "    if int(r['update'])<cut: continue\n",
         "    a=C[r['condition']][int(r['seed'])]\n",
         "    a[0]+=float(r['eat0']);a[1]+=float(r['eat1']);a[2]+=float(r['enc0']);a[3]+=float(r['enc1']);a[4]+=float(r['selectivity']);a[5]+=1\n",
         "o0={c:{s:C[c][s][0]/max(C[c][s][2],1e-9) for s in C[c]} for c in C}\n",
         "o1={c:{s:C[c][s][1]/max(C[c][s][3],1e-9) for s in C[c]} for c in C}\n",
         "def M(v): return np.mean(v),(np.std(v,ddof=1)/np.sqrt(len(v)) if len(v)>1 else float('nan'))\n",
         "def dd(mark,ctrl,cond):\n",
         "    ss=sorted(set(mark['none'])&set(mark[cond])); return M([(mark['none'][s]-mark[cond][s])-(ctrl['none'][s]-ctrl[cond][s]) for s in ss]), [(mark['none'][s]-mark[cond][s])-(ctrl['none'][s]-ctrl[cond][s]) for s in ss]\n",
         "print(f\"{'cond':>5} | {'opp0(pois)':>10} {'opp1(silly)':>11} {'sel':>5}\")\n",
         "for c in ('none','0','1'):\n",
         "    if c not in C: print(f'{c:>5} | (pending)'); continue\n",
         "    sl=np.mean([C[c][s][4]/C[c][s][5] for s in C[c]])\n",
         "    print(f'{c:>5} | {np.mean(list(o0[c].values())):10.3f} {np.mean(list(o1[c].values())):11.3f} {sl:5.2f}')\n",
         "if '0' in C: (m,se),_=dd(o0,o1,'0'); print(f'\\nPOISON none->(0,) diff-in-diff: {m:+.4f} margin {m/se:+.2f}')\n",
         "if '1' in C:\n",
         "    (m,se),ps=dd(o1,o0,'1'); v='SILLY INSTALLS (c_zapped=35)' if m/se>=2 else 'silly flat even at c=35'\n",
         "    print(f'SILLY  none->(1,) diff-in-diff: {m:+.4f} margin {m/se:+.2f}  <== {v}')\n",
         "    print('  per-seed:', ' '.join(f'{x:+.3f}' for x in ps))"),
]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
json.dump(nb, open("zap35_gpu.ipynb","w",encoding="utf-8"), indent=1)
print("wrote zap35_gpu.ipynb")
