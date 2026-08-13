"""Internalization probe -- turnkey arms (frozen + cue-masked) on the SURVIVING norms.
Per PROBE_PREREGISTRATION.md (predictions fixed before run). Flagship config, 5 seeds.
  frozen   : freeze_after=1000  -> grounded norm holds => in weights (corroborating)
  cue-mask : unmark_after=1000  -> grounded norm survives => NOT enforcement-cue-
             reconstructed (stored); collapses => cue-driven. THE behavioral discriminator.
Each notebook prints the ghost-end endpoint gap vs the normal persist value, so the
pre-registered verdict (survives / collapses) is read straight off. Regenerate:
python make_probe_gpu.py
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
    "def endpoint_verdict(path, cond, mi, nbt, mode, normal, lo=1500, hi=1600):\n",
    "    r=[x for x in _csv.DictReader(open(path)) if x['condition']==cond]\n",
    "    S=sorted(set(int(x['seed']) for x in r)); ctrl=max(t for t in range(nbt) if t!=mi)\n",
    "    def o(s,i):\n",
    "        rs=[x for x in r if int(x['seed'])==s and lo<=int(x['update'])<hi]\n",
    "        e=sum(float(x['eat'+str(i)]) for x in rs); c=sum(float(x['enc'+str(i)]) for x in rs)\n",
    "        return e/max(c,1e-9)\n",
    "    v=[(o(s,ctrl)-o(s,mi)) if mode=='avoid' else (o(s,mi)-o(s,ctrl)) for s in S]\n",
    "    m=_st.mean(v)\n",
    "    verdict='SURVIVES (>= half of normal) -> stored/not-cue-reconstructed' if m>0.5*normal else 'COLLAPSES (-> ~0) -> cue-driven/re-learned'\n",
    "    print('%s: ghost-end gap %+.3f (normal persist %+.3f) per-seed %s -> %s'%(path,m,normal,[round(x,3) for x in v],verdict))\n",
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

ENV_FULL = ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
            "                    n_berry_types=3, grid=22, ghost_keeps_bonus=False)\n")
ENV_COORD = ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
             "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
             "                    n_berry_types=3, grid=22, ghost_keeps_bonus=False,\n"
             "                    convergent_berry=1, coord_k=2.0, coord_a=1.5, conformity_berry=1)\n")

# (norm, env, cond, marked_idx, mode, normal-persist endpoint, analyze-converge)
NORMS = [("poison", ENV_FULL, "[(0,)]", "0", 0, "avoid", 0.234, ""),
         ("coord",  ENV_COORD, "[(1,)]", "1", 1, "converge", 0.203, ", converge={1}")]
ARMS = [("frozen", "freeze_after=1000", "grounded norm HOLDS under frozen weights => behavior is in the weights (corroborating; ground still live so not proof)"),
        ("cuemask", "unmark_after=1000", "grounded norm SURVIVES with the mark cue masked => NOT enforcement-cue-reconstructed => stored (THE discriminator); COLLAPSES => cue-driven")]
for nm, env, cond, cstr, mi, mode, normal, conv in NORMS:
    for arm, flag, pred in ARMS:
        build(f"probe_{nm}_{arm}_gpu.ipynb", f"probe_{nm}_{arm}.csv",
              [f"# Internalization probe: {nm} + {arm}\n", "\n",
               f"**Pre-registered prediction:** {pred}.\n", "\n",
               "Flagship config, 5 seeds. Ghost-end endpoint gap is compared to the normal "
               f"persist value ({normal:+.3f}); survives (>= half) = stored, collapses = cue/re-learned.\n"],
              [env, "hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
               f"extinct.run_extinction({cond}, ENV, hp, n_install=1000, n_seeds=5, {flag},\n",
               f"                       vmap_seeds=5, out_csv='probe_{nm}_{arm}.csv')\n",
               f"extinct.analyze('probe_{nm}_{arm}.csv'{conv})\n",
               f"endpoint_verdict('probe_{nm}_{arm}.csv', '{cstr}', {mi}, 3, '{mode}', {normal})"])
