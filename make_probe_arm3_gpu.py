"""Arm 3 (hidden-state decode + ablate) notebooks -> ./notebooks/, one per norm for 3 pods.
Predictions fixed in PROBE_PREREGISTRATION.md. Each notebook trains 5 flagship policies
(install+ghost, learning-on), then decodes the norm direction from the GRU carry and
ablates it. Saves per-seed results to JSON + prints the verdict. Regenerate:
python make_probe_arm3_gpu.py
"""
import json, base64, os
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py", "extinct.py", "probe_arm3.py"]
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
def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def _src():
    L=["import base64, pathlib\n","_SRC={\n"]
    for p in SRC:
        with open(p,"rb") as f: L.append("  %r: %r,\n"%(p, base64.b64encode(f.read()).decode()))
    L+=["}\n","for _n,_b in _SRC.items(): pathlib.Path(_n).write_bytes(base64.b64decode(_b))\n","print('wrote sources')\n"]
    return code(*L)
def build(nb, head, run):
    cells=[md(*head), md("## 0. Setup"), code(*SETUP), md("## 1. Sources"), _src(),
           md("## 2. Run (trains 5 flagship policies, then decode+ablate -- the long cell)"),
           code("print('>>> RUN START', flush=True)\n","import probe_arm3 as P, run_sweep as R, json\n",*run,
           "\n","print('>>> RUN DONE', flush=True)\n")]
    nbo={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},
         "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
    json.dump(nbo, open(os.path.join(OUTDIR,nb),"w",encoding="utf-8"), indent=1)
    print("wrote", os.path.join(OUTDIR,nb))

ENV_FULL = ("R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "              zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
            "              n_berry_types=3, grid=22, ghost_keeps_bonus=False)")
ENV_COORD = ("R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
             "              zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
             "              n_berry_types=3, grid=22, ghost_keeps_bonus=False,\n"
             "              convergent_berry=1, coord_k=2.0, coord_a=1.5, conformity_berry=1)")

# (norm, env, marked, mi, ci, mode, pre-registered prediction)
NORMS = [
 ("poison", ENV_FULL, "(0,)", 0, 2, "avoid",
  "DECODABLE (high R2) + CAUSAL, but partly mark-conditioned (cue-mask was 46%): ablation moves the cue-independent half."),
 ("coord", ENV_COORD, "(1,)", 1, 2, "converge",
  "DECODABLE + cue-independent (cue-mask 103%) AND adaptation-sensitive (frozen 76%): ablation moves behavior; direction may be less stable than poison's."),
 ("vestige", ENV_FULL, "(1,)", 1, 2, "avoid",
  "R2 ~ CHANCE (norm reverted at ghost-end) + ablation INERT: nothing stored to decode or move."),
]
for nm, env, marked, mi, ci, mode, pred in NORMS:
    build(f"probe_arm3_{nm}_gpu.ipynb",
          [f"# Arm 3 decode+ablate: {nm}\n", "\n",
           f"**Pre-registered prediction:** {pred}\n", "\n",
           "5 flagship seeds. Decodability R2 = norm behavior linearly read from the GRU "
           "carry; ablation moves the norm gap; random-direction ablation = control.\n"],
          [f"ENV = {env}\n",
           "hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
           "res=[]\n",
           "for s in range(5):\n",
           f"    r = P.run(ENV, hp, n_install=1000, N=12, marked={marked}, mi={mi}, ci={ci}, mode='{mode}', seed=s, E=128)\n",
           "    res.append(r); print('seed', s, {k: round(v,3) for k,v in r.items()}, flush=True)\n",
           f"json.dump(res, open('probe_arm3_{nm}.json','w'))\n",
           f"P.report('{nm}', res)"])
