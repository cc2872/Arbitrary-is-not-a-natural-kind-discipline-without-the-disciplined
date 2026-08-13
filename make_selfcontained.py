"""Generate pop_sweep_selfcontained.ipynb -- a SELF-CONTAINED Phase-2 pilot #1
notebook. It embeds the current .py sources via %%writefile cells, so there is
nothing to upload: open on Colab (or locally), set GPU, Run all.

Because it reads the .py files at generation time, the notebook can never drift
from the committed source (the failure mode we hit with the old hand-edited
self-contained notebook). Regenerate after any .py edit:

    python make_selfcontained.py
"""
import json

# The import chain pilot #1 needs, in dependency order.
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py",
       "run_sweep.py", "pop_sweep.py"]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": list(lines)}


def writefile_cell(path):
    """A code cell that rewrites `path` on disk from the current source."""
    with open(path, encoding="utf-8") as f:
        body = f.read()
    # json.dump handles all escaping; keep line structure for a readable diff.
    return code(f"%%writefile {path}\n", *body.splitlines(keepends=True))


cells = [
    md("# Phase 2 - Pilot #1: does the silly-rule norm survive at smaller N?\n",
       "\n",
       "**Self-contained.** Every code file is written to disk by the "
       "`%%writefile` cells below - nothing to upload. Set `Runtime -> Change "
       "runtime type -> GPU`, then `Runtime -> Run all`.\n",
       "\n",
       "**The question.** Gate 1 (Koster replication) cleared at N=12. The "
       "extinction experiment was speced at N=6, but the silly-rule effect grows "
       "with population, so N=6 is a real risk. This pilot sweeps "
       "`N in {6,8,10,12}` at the *tuned replication config* (D=100, "
       "bonus=8.75, 800 updates, 5 seeds) and reports the **paired** berry-1 "
       "norm signal `d1 = eat1[(0,)] - eat1[(0,1)]`. The lowest N where `d1` "
       "clears margin>=2 is the smallest population with a norm to extinguish - "
       "that decision sets the extinction N and unblocks the registration.\n",
       "\n",
       "**Config caveat:** the replication's `num_envs` was never recorded and it "
       "changes the outcome. Run **section 2.5 first** to recover it; the N-sweep "
       "is only valid once its N=12 cell matches the replication.\n",
       "\n",
       "Runs crash-safe (per-cell CSV writes), so `--analyze pop_sweep.csv` "
       "works on a partial run."),

    md("## 0. Environment (never pins numpy/scipy - that was the old failure)"),
    code("# jax is preinstalled on Colab GPU; only add flax/optax if missing.\n",
         "# NEVER uninstall or version-pin numpy/scipy: it corrupts the running\n",
         "# kernel and cascades into every later cell.\n",
         "import importlib, subprocess, sys\n",
         "missing = [m for m in ('flax', 'optax') if importlib.util.find_spec(m) is None]\n",
         "if missing:\n",
         "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=False)\n",
         "import jax, flax, optax, numpy as np\n",
         "print('jax', jax.__version__, ' flax', flax.__version__, ' optax', optax.__version__)\n",
         "devs = jax.devices(); print('devices:', devs)\n",
         "if not any(d.platform == 'gpu' for d in devs):\n",
         "    print('\\n[!] No GPU visible. Runtime -> Change runtime type -> GPU,'\n",
         "          ' then Run all. (The tiny sanity cell still runs on CPU; the'\n",
         "          ' full --run will abort on CPU by design.)')\n",
         "else:\n",
         "    print('\\nOK - GPU ready.')"),

    md("## 1. Write the source files (self-contained; regenerated from the repo)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run (allowed on CPU, ~1-2 min)\n",
       "Confirms the whole path executes and the paired readout prints before you "
       "spend GPU time. Meaningless numbers (n=2, 20 updates) - this is plumbing."),
    code("!python pop_sweep.py --smoke"),

    md("## 2.5 RECOVER THE REPLICATION REGIME (run this first)\n",
       "The `num_envs` behind koster_replication.csv was never recorded, and it "
       "flips the result: at `num_envs=64` even the no-rule `none` condition "
       "learns poison avoidance unaided, so the taboo is redundant and the effect "
       "vanishes (a false null). This sweeps `num_envs in {16,32,48,64}` at N=12 "
       "and finds the budget where the replication dissociation returns: `none` "
       "eat0 ~90 (fails) while `(0,1)` ~45 (avoids), i.e. `none-(0,1) ~ 45`. "
       "**Use that num_envs for the N-sweep below.**"),
    code("!python pop_sweep.py --recover"),

    md("## 3. Pilot #1 - the real sweep (GPU)\n",
       "`N in {6,8,10,12}` x 3 conditions x 5 seeds x 800 updates. ~3x the "
       "replication run's wall-clock (small-N cells are cheaper). Aborts in <1s "
       "on CPU by design - run this on GPU."),
    code("!python pop_sweep.py --run"),

    md("## 4. Read the verdict (re-runnable anytime, even mid-run)\n",
       "Lowest N with paired `d1` margin>=2 = smallest population where a "
       "socially-grounded norm exists to extinguish."),
    code("!python pop_sweep.py --analyze pop_sweep.csv"),

    code("# save the result\n",
         "try:\n",
         "    from google.colab import files; files.download('pop_sweep.csv')\n",
         "except Exception as e:\n",
         "    print('not on Colab / nothing to download:', e)"),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("pop_sweep_selfcontained.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote pop_sweep_selfcontained.ipynb  (embedded:", ", ".join(SRC) + ")")
