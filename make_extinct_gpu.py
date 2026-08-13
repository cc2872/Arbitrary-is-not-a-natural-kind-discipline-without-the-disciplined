"""Generate extinct_gpu.ipynb -- the extinction experiment on a rented A100/H100.
Installs the mark-contingent norm under enforcement (updates < n_install), then
flips enforce=False (ghost) and logs the per-update opportunity curve to see
whether acquired avoidance DECAYS (extinction) or HOLDS (persistence).

Poison (intrinsic penalty) is expected to persist -> the physically-anchored
control. A purely-social silly-rule norm, IF it installed (see oppprobe), is the
one whose decay is the flagship result. Runs all 3 conditions at bonus 8.75.

CAVEAT baked into the readout: the ghost cell leaves other agents visibly
behaving, so a decayed-taboo result is still conformity-confounded until the
full-population-removal knockout is run. This is the machinery + first curve.

Regenerate:  python make_extinct_gpu.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py", "extinct.py"]


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}
def writefile_cell(p):
    with open(p, encoding="utf-8") as f: body = f.read()
    return code(f"%%writefile {p}\n", *body.splitlines(keepends=True))


cells = [
    md("# Extinction experiment (rented A100/H100): does the norm decay when enforcement is removed?\n",
       "\n",
       "Two phases in one training run: **install** the mark-contingent norm under "
       "enforcement (updates < n_install), then flip `enforce=False` (ghost cell — "
       "target penalty + removal gone, zapper cost/beam kept) and keep training. "
       "The per-update opportunity metric `opp_t = eat_t/enc_t` traces whether the "
       "acquired avoidance **climbs back (extinction)** or **holds (persistence)**.\n",
       "\n",
       "- **Poison (berry-0):** intrinsic delayed penalty → expected to *persist* "
       "→ the physically-anchored control.\n",
       "- **Silly (berry-1):** purely social → *if it installed*, its decay is the "
       "flagship signal.\n",
       "\n",
       "**Caveat (in the readout):** the ghost cell leaves other agents visibly "
       "behaving, so a decayed taboo is still conformity-confounded until a "
       "full-population-removal knockout is run. This is the machinery + the first "
       "decay curve, not yet the clean flagship."),

    md("## 0. Install JAX-GPU + flax + optax"),
    code("!pip install -q -U \"jax[cuda12]\" flax optax\n",
         "import jax; dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', 'No GPU - match jax[cudaXX] to the image CUDA.'\n",
         "print('GPU OK:', dev[0].device_kind)"),

    md("## 1. Write the source files (extinction schedule in train_jax + extinct.py)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny extinction (confirms the enforce flag flips mid-run)\n",
       "Run IN-KERNEL (not `!python`) so it shares the kernel's GPU allocation "
       "instead of a second process fighting it for memory."),
    code("import run_sweep as R, extinct\n",
         "ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
         "                    zap_removal_steps=25, bonus_requires_mark=True)\n",
         "hp = dict(R.FAITHFUL_HP); hp['num_envs'] = 16; hp['updates'] = 30\n",
         "extinct.run_extinction([(0, 1)], ENV, hp, n_install=15, n_seeds=1,\n",
         "                       out_csv='extinct_smoke.csv')\n",
         "extinct.analyze('extinct_smoke.csv')"),

    md("## 3. Run the extinction experiment (faithful, IN-KERNEL)\n",
       "Conditions `(0,)` poison-only, `(1,)` silly-only, `()` baseline × 5 seeds, "
       "1000 updates install (enforce ON) + 600 extinction (enforce OFF) = 1600, "
       "bonus 8.75 mark-contingent. Sequential seeds, in-kernel (one process → no "
       "preallocation clash). Heavy (~a few hours). `extinct.csv` crash-safe per condition."),
    code("import run_sweep as R, extinct\n",
         "ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
         "                    zap_removal_steps=25, bonus_requires_mark=True)\n",
         "hp = dict(R.FAITHFUL_HP); hp['updates'] = 1600\n",
         "extinct.run_extinction([(), (0,), (1,)], ENV, hp, n_install=1000,\n",
         "                       out_csv='extinct.csv')\n",
         "extinct.analyze('extinct.csv')"),

    md("## 4. Re-read the decay curves (RELATIVE frame)\n",
       "Per condition, the marked berry's rise across the enforce=False switch "
       "MINUS its unmarked counterpart's drift. Silly `(1,)` DECAYS while poison "
       "`(0,)` PERSISTS = the flagship dissociation."),
    code("import extinct; extinct.analyze('extinct.csv')"),

    md("## 5. Outputs"),
    code("import glob, os\n",
         "for f in sorted(glob.glob('extinct*.csv')):\n",
         "    print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open("extinct_gpu.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote extinct_gpu.ipynb  (embedded:", ", ".join(SRC) + ")")
