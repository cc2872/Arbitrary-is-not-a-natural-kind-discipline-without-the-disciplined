"""EMERGENCE study: does a coordination convention EMERGE (symmetry-breaking) rather than
being designated? World = 4 berries; berry 0 = poison (avoided outside option), berries
1,2,3 = SYMMETRIC convention candidates (each pays coord_k*n_j^1.5 for co-eating itself).
No enforcement (r_zap_bonus=0, no marks) -> pure self-organization. Which berry becomes
the convention is endogenous and differs across seeds = the emergence proof.

Wave 1 (calibration, one per GPU): coord_k in {0, 0.5, 1.0, 2.0}. Read the concentration
(max-share across candidates: ~0.33 = no convention, ->1 = converged) and whether the
winning berry differs across seeds. Pick the lowest coord_k that reliably converges.
Regenerate: python make_emergence_gpu.py
"""
import json, base64, os
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py", "extinct.py"]
OUTDIR = "notebooks"; os.makedirs(OUTDIR, exist_ok=True)
SETUP = [
    "import os, sys, subprocess\n",
    "os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')\n",
    "def _pip(*p): subprocess.run([sys.executable,'-m','pip','install','-q',*p])\n",
    "try:\n", "    import jax, flax, optax\n", "    on_gpu = jax.devices()[0].platform=='gpu'\n",
    "except Exception:\n", "    on_gpu = False\n",
    "if not on_gpu:\n", "    print('No GPU-bound jax -> installing jax[cuda12]...', flush=True)\n",
    "    _pip('jax[cuda12]','flax','optax')\n",
    "    raise SystemExit('*** Installed. Kernel > Restart, then Run All. ***')\n",
    "import jax\n", "print('jax', jax.__version__, '| devices:', jax.devices())\n",
]
READOUT = [
    "import csv as _csv, statistics as _st\n",
    "def emergence_readout(path, candidates=(1,2,3)):\n",
    "    rows=list(_csv.DictReader(open(path))); seeds=sorted(set(int(r['seed']) for r in rows))\n",
    "    umax=max(int(r['update']) for r in rows); lo=umax-99\n",
    "    winners=[]; concs=[]\n",
    "    for s in seeds:\n",
    "        rs=[r for r in rows if int(r['seed'])==s and int(r['update'])>=lo]\n",
    "        eats={j: sum(float(r['eat'+str(j)]) for r in rs) for j in candidates}\n",
    "        tot=sum(eats.values()) or 1e-9; sh={j: eats[j]/tot for j in candidates}\n",
    "        w=max(sh,key=sh.get); winners.append(w); concs.append(round(sh[w],2))\n",
    "    base=1.0/len(candidates)\n",
    "    verdict='CONVENTION EMERGED' if _st.mean(concs)>0.6 else ('partial' if _st.mean(concs)>0.45 else 'NO convention (spread)')\n",
    "    print('%s: mean max-share %.2f  (%.2f=no convention -> 1=converged)  -> %s'%(path,_st.mean(concs),base,verdict))\n",
    "    print('  per-seed WINNER berry: %s  (differ across seeds => EMERGENT symmetry-breaking)'%winners)\n",
    "    print('  per-seed max-share:    %s'%concs)\n",
]
def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def _src():
    L=["import base64, pathlib\n","_SRC={\n"]
    for p in SRC:
        with open(p,"rb") as f: L.append("  %r: %r,\n"%(p, base64.b64encode(f.read()).decode()))
    L+=["}\n","for _n,_b in _SRC.items(): pathlib.Path(_n).write_bytes(base64.b64decode(_b))\n","print('wrote sources')\n"]
    return code(*L)
def build(nb, csvf, head, run):
    cells=[md(*head), md("## 0. Setup"), code(*SETUP), md("## 1. Sources"), _src(),
           md("## 2. Run"), code("print('>>> RUN START', flush=True)\n","import run_sweep as R, extinct\n",
           *READOUT, *run, "\n","print('>>> RUN DONE', flush=True)\n")]
    nbo={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},
         "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
    json.dump(nbo, open(os.path.join(OUTDIR,nb),"w",encoding="utf-8"), indent=1)
    print("wrote", os.path.join(OUTDIR,nb), "->", csvf)

def env(k):   # 4 berries; berry0=poison outside option; 1,2,3 symmetric candidates; NO enforcement
    return ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=0.0, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
            "                    n_berry_types=4, grid=26,\n"
            f"                    convention_berries=(1,2,3), coord_k={k}, coord_a=1.5)\n")

# WAVE 1 -- calibration: does a convention emerge, and at what coord_k?
for k, ktag in [(0.0, "0"), (0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0"), (4.0, "4p0"), (8.0, "8p0")]:
    lbl = "coord_k=0 CONTROL (expect NO convention, spread ~0.33)" if k == 0 else f"coord_k={k}"
    build(f"emerge_k{ktag}_gpu.ipynb", f"emerge_k{ktag}.csv",
          [f"# Emergence calibration: {lbl}\n", "\n",
           "4-berry world (0=poison outside option; 1,2,3 = SYMMETRIC convention candidates, "
           "each paying coord_k*n^1.5 for co-eating itself). NO enforcement. Does a convention "
           "self-organize? Read: mean max-share (0.33=no convention -> 1=converged) and whether "
           "the WINNING berry differs across seeds (that difference IS the emergence proof).\n"],
          [env(k), "hp = dict(R.FAITHFUL_HP); hp['updates']=1200\n",
           f"extinct.run_extinction([()], ENV, hp, n_install=1200, n_seeds=5, vmap_seeds=5,\n",
           f"                       out_csv='emerge_k{ktag}.csv')\n",
           f"emergence_readout('emerge_k{ktag}.csv')"])
