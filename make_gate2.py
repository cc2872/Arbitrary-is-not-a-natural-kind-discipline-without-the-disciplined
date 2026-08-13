"""Generate gate_2.ipynb -- self-contained Phase-2 (Gate 2) setup notebook.

Rebuilt (2026-08-01) to match the conditions that produced koster_replication.csv
and recover the one unrecorded knob. Forensic audit fixed everything else:

    koster_replication.csv = run_sweep([12], [(),(0,),(0,1)], n_seeds=5,
        hp=FAITHFUL_HP+updates=800,
        env=env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,
                        zap_removal_steps=25))          # pending HIDDEN
        on Colab GPU, code = berryworld_jax@a954774 + train_jax@d2b0ee1.

git proves env code, train_jax, and every named FAITHFUL_HP value are identical
between koster (none->91) and confirm256 (none->43); observe_pending is ruled out
(a954774 already zeroed pending). The sole unrecorded run-level knob is num_envs.
So section 3 sweeps num_envs at the exact koster grid to find the value that
reproduces none->91; section 4 runs the population pilot there. If NO num_envs
recovers it, the 91 is a session/GPU bistability artifact, not a config -- stop
and reconsider whether koster can anchor the replication.

Embeds the current .py sources via %%writefile (nothing to upload, no drift).
Regenerate after any .py edit:  python make_gate2.py
"""
import json

SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py",
       "run_sweep.py", "pop_sweep.py"]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": list(lines)}


def writefile_cell(path):
    with open(path, encoding="utf-8") as f:
        body = f.read()
    return code(f"%%writefile {path}\n", *body.splitlines(keepends=True))


cells = [
    md("# Gate 2 - recover the koster config, then pick the extinction N\n",
       "\n",
       "**Self-contained.** All code is written by the `%%writefile` cells below "
       "- nothing to upload. `Runtime -> Change runtime type -> GPU`, then run top "
       "to bottom, **stopping at the decision in section 3**.\n",
       "\n",
       "**The problem.** `koster_replication.csv` (the trustworthy dissociation: "
       "`none` eat0 ~91, `(0,1)` ~45, all 5 seeds tight, reproduced twice) does "
       "not reproduce with the current code at the config we can specify - "
       "`confirm256` gives `none` ~43 (learns unaided, no dissociation).\n",
       "\n",
       "**What the forensic audit fixed.** From the CSV columns + git, koster ran "
       "`run_sweep([12], [(),(0,),(0,1)], n_seeds=5, hp=FAITHFUL_HP+updates=800, "
       "env=env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300, "
       "zap_removal_steps=25))`. git shows the env code, `train_jax`, and every "
       "named `FAITHFUL_HP` value are **identical** to today's, and `observe_pending` "
       "is ruled out (the code that made koster already zeroed pending). The **one** "
       "run-level knob the CSV never recorded is `num_envs` - and `run_sweep.py`'s "
       "own note says *num_envs was the difference that flipped the whole result*.\n",
       "\n",
       "**This notebook.** Section 3 sweeps `num_envs` at the exact koster grid and "
       "reports which value reproduces `none->91`. Section 4 runs the population "
       "pilot at that recovered value. If none recover, `none->91` is a session/GPU "
       "artifact - stop. Crash-safe per cell."),

    md("## 0. Environment (never pins numpy/scipy)"),
    code("import importlib, subprocess, sys\n",
         "missing = [m for m in ('flax', 'optax') if importlib.util.find_spec(m) is None]\n",
         "if missing:\n",
         "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=False)\n",
         "import jax, flax, optax, numpy as np\n",
         "print('jax', jax.__version__, ' flax', flax.__version__, ' optax', optax.__version__)\n",
         "devs = jax.devices(); print('devices:', devs)\n",
         "print('OK - GPU ready.' if any(d.platform == 'gpu' for d in devs)\n",
         "      else '[!] No GPU. Runtime -> Change runtime type -> GPU, then Run all.')"),

    md("## 1. Write the source files (self-contained, regenerated from the repo)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run (CPU-allowed, ~1-2 min)\n",
       "Confirms the path executes end to end. Numbers are meaningless "
       "(n=2, 20 updates)."),
    code("!python pop_sweep.py --smoke"),

    md("## 3. RECOVER THE KOSTER CONFIG (decision gate - read the verdict)\n",
       "The exact koster grid at N=12 (D=100, bonus=8.75, ep=300, removal=25, "
       "**pending hidden**, 5 seeds, 800 updates), swept over `num_envs`. The "
       "koster signature is `none` eat0 ~91 (never learns to avoid poison unaided) "
       "and `none - (0,1)` ~45.\n",
       "\n",
       "- **some num_envs -> none ~91**: that value is the recovered koster config; "
       "section 4 runs the pilot there.\n",
       "- **every num_envs -> none ~43**: koster's dissociation is not reproducible "
       "from `num_envs`. It is then a session/GPU bistability artifact (rerun this "
       "cell on a **fresh runtime** - Runtime -> Disconnect and delete - to see if "
       "the same value flips). Until it reproduces, koster cannot anchor the "
       "replication - **stop, do not run section 4.**"),
    code("import run_sweep as R, numpy as np\n",
         "\n",
         "# exact koster env: pending hidden (observe_pending defaults False)\n",
         "KOSTER_ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75,\n",
         "                           episode_len=300, zap_removal_steps=25)\n",
         "NE_GRID = [64, 128, 256, 512]     # 64/256 = known-null anchors; 128/512 untested\n",
         "\n",
         "print('num_envs recovery at the koster grid '\n",
         "      '(N=12, D=100, bonus=8.75, 5 seeds, 800 updates, pending hidden)')\n",
         "print('koster signature = none eat0 ~91 and none-(0,1) ~45.\\n')\n",
         "print(f\"{'num_envs':>8} | {'none':>6} {'(0,)':>6} {'(0,1)':>6} | \"\n",
         "      f\"{'none-(0,1)':>11} | recovers?\")\n",
         "print('-' * 60)\n",
         "best_ne = None\n",
         "for ne in NE_GRID:\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = 800; hp['num_envs'] = ne\n",
         "    rows = R.run_sweep([12], [(), (0,), (0, 1)], n_seeds=5, hp=hp,\n",
         "                       env_list=[KOSTER_ENV], out_csv=f'recover_ne{ne}.csv')\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(0.1 * (umax + 1))\n",
         "    e = {c: np.mean([float(r['eat0']) for r in rows\n",
         "                     if r['condition'] == c and int(r['update']) >= cut])\n",
         "         for c in ('none', '0', '01')}\n",
         "    diss = e['none'] - e['01']; rec = e['none'] > 70 and diss > 25\n",
         "    if rec and best_ne is None: best_ne = ne\n",
         "    print(f\"{ne:>8} | {e['none']:>6.1f} {e['0']:>6.1f} {e['01']:>6.1f} | \"\n",
         "          f\"{diss:>11.1f} | {'YES' if rec else 'no'}\")\n",
         "print()\n",
         "if best_ne is not None:\n",
         "    print(f'>>> RECOVERED: num_envs={best_ne} reproduces the koster dissociation.'\n",
         "          f' Section 4 runs the pilot at num_envs={best_ne}.')\n",
         "else:\n",
         "    print('>>> NOT RECOVERED: every num_envs gives none~43 (learns unaided).')\n",
         "    print('    koster none->91 is NOT config-reproducible here. Try a higher num_envs')\n",
         "    print('    (1024), or rerun on a FRESH runtime to test session/GPU bistability.')\n",
         "    print('    Until it reproduces, koster cannot anchor the replication -- STOP.')"),

    md("## 4. Population pilot at the recovered config (only if section 3 RECOVERED)\n",
       "`N in {6,8,10,12}` x 3 conditions x 5 seeds at the recovered `num_envs`, "
       "pending hidden. Writes `pop_sweep_ne<best_ne>.csv`. Uses `best_ne` from "
       "section 3 - if that section did not recover, this cell refuses to run."),
    code("import pop_sweep as P\n",
         "if best_ne is None:\n",
         "    print('Section 3 found no recovering num_envs -- do not run the pilot.'\n",
         "          ' Resolve section 3 first.')\n",
         "else:\n",
         "    print(f'Population pilot at recovered config: num_envs={best_ne}, '\n",
         "          f'pending hidden, N in {P.N_GRID}\\n')\n",
         "    P.run(num_envs=best_ne)\n",
         "    P.analyze(f'pop_sweep_ne{best_ne}.csv')"),

    md("## 5. Verdict - lowest N with a norm to extinguish\n",
       "Paired berry-1 signal `d1 = eat1[(0,)] - eat1[(0,1)]`; SURVIVES at "
       "margin>=2. Lowest surviving N sets the extinction population. The "
       "compliance table under it is the regime check - at the recovered config "
       "`none` eat0 should be ~90, not ~43. (Re-runnable mid-run on the partial CSV.)"),
    code("import pop_sweep as P\n",
         "if best_ne is not None:\n",
         "    P.analyze(f'pop_sweep_ne{best_ne}.csv')\n",
         "else:\n",
         "    print('No recovered config to analyze.')"),

    code("# save every recovery + pilot CSV\n",
         "try:\n",
         "    from google.colab import files, drive\n",
         "    import glob\n",
         "    for fpath in sorted(glob.glob('recover_ne*.csv') + glob.glob('pop_sweep_ne*.csv')):\n",
         "        files.download(fpath)\n",
         "except Exception as ex:\n",
         "    print('not on Colab / nothing to download:', ex)"),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("gate_2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote gate_2.ipynb  (embedded:", ", ".join(SRC) + ")")
