"""Generality batch for NMI: replicate the 3-row taxonomy at a 2nd population size
(N=24) and a 2nd environment (4-berry world). Direction is the claim (magnitudes are
config-dependent). 6 notebooks -> ./notebooks/, two waves of 3 (one per GPU). Same
3-cell/no-shell/base64/self-heal format. Regenerate: python make_generality_gpu.py

Rows per config:
  ghost      -> silly (cond 1,) DECAYS + poison (cond 0,) PERSISTS   [vestige + environmental]
  hazardoff  -> poison (cond 0,) + hazard_off_after=1000 DECAYS      [environmental own-knockout]
  coord      -> coordination cleanghost PERSISTS + flat DECAYS       [coordination row]
coord_k=2.0 inherited from the N=12 tuning; the coord INSTALL phase is the per-config fuse
(if opp1 isn't driven clearly above the companion, bump coord_k for that config and rerun).
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
def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def _src():
    L = ["import base64, pathlib\n", "_SRC = {\n"]
    for p in SRC:
        with open(p, "rb") as f: L.append("  %r: %r,\n" % (p, base64.b64encode(f.read()).decode()))
    L += ["}\n", "for _n,_b in _SRC.items(): pathlib.Path(_n).write_bytes(base64.b64decode(_b))\n",
          "print('wrote sources')\n"]
    return code(*L)
def build(nb, csv, head, run):
    cells = [md(*head), md("## 0. Setup"), code(*SETUP), md("## 1. Sources"), _src(),
             md("## 2. Run"), code("print('>>> RUN START', flush=True)\n",
             "import run_sweep as R, extinct\n", *run, "\n", "print('>>> RUN DONE', flush=True)\n")]
    nbo = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
           "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nbo, open(os.path.join(OUTDIR, nb), "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.join(OUTDIR, nb), "->", csv)

# env-line builders (grid/berry-count per config); coordination adds convergent+conformity
def env_full(nbt, grid):
    return ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
            f"                    n_berry_types={nbt}, grid={grid}, ghost_keeps_bonus=False)\n")
def env_coord(nbt, grid):
    return ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
            f"                    n_berry_types={nbt}, grid={grid}, ghost_keeps_bonus=False,\n"
            "                    convergent_berry=1, coord_k=2.0, coord_a=1.5, conformity_berry=1)\n")

def env_coord_k(nbt, grid, ck):   # coordination env with an explicit coord_k
    return ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True, c_zapped=2.0,\n"
            f"                    n_berry_types={nbt}, grid={grid}, ghost_keeps_bonus=False,\n"
            f"                    convergent_berry=1, coord_k={ck}, coord_a=1.5, conformity_berry=1)\n")

# ONE run per notebook (N=24 crashed on the 2nd run in a shared kernel: 2x agents ->
# the 1st run's cached graph doesn't free -> 2nd OOMs). coord_k rescaled per config:
# bonus ~ coord_k * n_coeat^1.5, so 2x agents (N=24) needs coord_k ~ 2.0*(1/2)^1.5 ~ 0.7.
# num_envs per config: N=24 uses 128 so its memory == the working N=12/256 runs
# (24*128 == 12*256); num_envs is regime-equivalent (64-512), so this is a pure memory
# fix. 4-berry stays at 256 (N=12 agent count). vmap_seeds also lowered for N=24.
CONFIGS = [   # tag, N, n_berry_types, grid, coord_k, vmap, num_envs, label
    ("N24", 24, 3, 22, 0.25, 2, 128, "N=24 (2x pop; num_envs=128 mem-matched to N=12/256; coord_k=0.25)"),
    ("4b",  12, 4, 26, 2.0, 5, 256, "4-berry world (2nd environment; coord_k=2.0 as N=12)"),
]
for tag, N, nbt, grid, ck, vmap, nenv, lbl in CONFIGS:
    hp_line = f"hp = dict(R.FAITHFUL_HP); hp['updates']=1600; hp['num_envs']={nenv}\n"
    for cond, cell, extra in [("[(1,)]", "silly", ""), ("[(0,)]", "poison", "")]:
        ns = 10 if (tag == "N24" and cell == "poison") else 5   # firm the frayed N24 poison cell
        build(f"gen_{tag}_{cell}_gpu.ipynb", f"gen_{tag}_{cell}.csv",
              [f"# Generality [{lbl}] -- {cell} ghost ({ns} seeds)\n", "\n",
               f"{cell} should {'DECAY' if cell=='silly' else 'PERSIST'} (same as N=12). "
               f"{'10 seeds: firm the frayed N24 persist cell with a saved, verifiable record.' if ns==10 else ''}\n"],
              [env_full(nbt, grid), hp_line,
               f"extinct.run_extinction({cond}, ENV, hp, n_install=1000, n_seeds={ns}, N={N},\n",
               f"                       vmap_seeds={vmap}, out_csv='gen_{tag}_{cell}.csv')\n",
               f"extinct.analyze('gen_{tag}_{cell}.csv')"])
    build(f"gen_{tag}_hazardoff_gpu.ipynb", f"gen_{tag}_hazardoff.csv",
          [f"# Generality [{lbl}] -- environmental own-knockout (hazard-off)\n", "\n",
           "poison (0,) + hazard_off_after=1000 should DECAY.\n"],
          [env_full(nbt, grid), hp_line,
           f"extinct.run_extinction([(0,)], ENV, hp, n_install=1000, n_seeds=5, N={N},\n",
           f"                       hazard_off_after=1000, vmap_seeds={vmap}, out_csv='gen_{tag}_hazardoff.csv')\n",
           f"extinct.analyze('gen_{tag}_hazardoff.csv')"])
    for cell, extra in [("coord_cleanghost", ""),
                        ("coord_flat", "                       flatten_returns_after=1000,\n")]:
        build(f"gen_{tag}_{cell}_gpu.ipynb", f"gen_{tag}_{cell}.csv",
              [f"# Generality [{lbl}] -- {cell}\n", "\n",
               f"{'cleanghost should PERSIST' if 'clean' in cell else 'flat should DECAY'}. "
               f"coord_k={ck}. Check install: opp1 clearly above the companion; adjust coord_k if not.\n"],
              [env_coord_k(nbt, grid, ck), hp_line,
               f"extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=5, N={N},\n",
               extra + f"                       vmap_seeds={vmap}, out_csv='gen_{tag}_{cell}.csv')\n",
               f"extinct.analyze('gen_{tag}_{cell}.csv', converge={{1}})"])
