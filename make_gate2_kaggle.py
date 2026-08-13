"""Generate gate_2_kaggle.ipynb -- the Gate-2 pipeline (num_envs recovery ->
pilot -> verdict) as a SINGLE self-contained Kaggle notebook you run top to
bottom. Same logic as gate_2.ipynb (Colab), Kaggle-flavored:
  * JAX+GPU is preinstalled on Kaggle -> install only flax+optax (no numpy churn)
  * no google.colab.files -> CSVs land in /kaggle/working/ (Output tab), no
    upload/download plumbing
  * sequential seeds are the run_sweep default (embedded) -> fits Kaggle's 16 GB

Embeds the current .py sources via %%writefile (nothing to upload, no drift).
Regenerate after any .py edit:  python make_gate2_kaggle.py
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
    md("# Gate 2 (Kaggle) - recover the koster num_envs, then pick the extinction N\n",
       "\n",
       "**Before running:** right sidebar -> **Session options -> Accelerator -> "
       "GPU** (P100 or T4 x2). Then **Run All** (or Save Version -> Save & Run All "
       "for a detached 12 h batch run).\n",
       "\n",
       "Self-contained: every code file is written by the `%%writefile` cells - "
       "nothing to upload. All CSVs land in `/kaggle/working/` (right sidebar -> "
       "**Output**). Seeds run **sequentially** (memory-safe on 16 GB).\n",
       "\n",
       "**What it does.** `koster_replication.csv` (none eat0 ~91, (0,1) ~45) does "
       "not reproduce at the config we can specify; the audit showed the env code, "
       "`train_jax`, and every named hyperparameter are identical, so the one "
       "unrecorded knob is `num_envs`. Section 3 sweeps `num_envs` at the exact "
       "koster grid and reports which value (if any) reproduces `none->91`; section "
       "4 runs the population pilot there. If nothing recovers, `none->91` is a "
       "session/GPU artifact - stop.\n",
       "\n",
       "*Runtime note:* the full sweep is heavy. Shrink `NE_GRID` (section 3) to "
       "fit; every num_envs writes its own `recover_ne<NE>.csv`, so partial runs "
       "still bank."),

    md("## 0. Install (flax+optax only; Kaggle's JAX already uses the GPU)"),
    code("!pip install -q flax optax\n",
         "import jax, numpy as np\n",
         "dev = jax.devices(); print('devices:', dev)\n",
         "assert dev[0].platform == 'gpu', (\n",
         "    'Still on CPU! Right sidebar -> Session options -> Accelerator -> GPU, then re-run.')\n",
         "print('GPU OK')"),

    md("## 1. Write the source files (self-contained; sequential seeds -> ~5x less VRAM)"),
    *[writefile_cell(p) for p in SRC],

    md("## 2. Sanity - tiny run (fast)\n",
       "Confirms the path executes end to end. Numbers are meaningless "
       "(n=2, 20 updates)."),
    code("!python pop_sweep.py --smoke"),

    md("## 3. RECOVER THE KOSTER CONFIG (decision gate - read the verdict)\n",
       "Exact koster grid at N=12 (D=100, bonus=8.75, ep=300, removal=25, "
       "**pending hidden**, 5 seeds, 800 updates), swept over `num_envs`. koster "
       "signature = `none` eat0 ~91 and `none - (0,1)` ~45.\n",
       "\n",
       "- **some num_envs -> none ~91**: recovered config; section 4 runs the "
       "pilot there.\n",
       "- **every num_envs -> none ~43**: not reproducible from `num_envs` "
       "(session/GPU artifact) - **stop, do not run section 4.**"),
    code("import run_sweep as R, numpy as np\n",
         "\n",
         "NE_GRID = [64, 128, 256, 512]     # shrink to fit your session if needed\n",
         "KOSTER_ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75,\n",
         "                           episode_len=300, zap_removal_steps=25)  # pending hidden\n",
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
         "    print('    koster none->91 is NOT config-reproducible here (session/GPU artifact).')\n",
         "    print('    Do not run section 4.')"),

    md("## 4. Population pilot at the recovered config (only if section 3 RECOVERED)\n",
       "`N in {6,8,10,12}` x 3 conditions x 5 seeds at the recovered `num_envs`, "
       "pending hidden. Writes `pop_sweep_ne<best_ne>.csv`. Refuses to run if "
       "section 3 recovered nothing."),
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
       "margin>=2. Lowest surviving N sets the extinction population.\n",
       "\n",
       "*Caveat:* `d1` is an **absolute** eat-count, which confounds a norm effect "
       "with foraging volume. A SURVIVES here is necessary but not sufficient - "
       "confirm it as a per-encounter **rate** (berry-1 share) before trusting it."),
    code("import pop_sweep as P\n",
         "if best_ne is not None:\n",
         "    P.analyze(f'pop_sweep_ne{best_ne}.csv')\n",
         "else:\n",
         "    print('No recovered config to analyze.')"),

    md("## 6. Outputs\n",
       "All CSVs (`recover_ne*.csv`, `pop_sweep_ne*.csv`) are in `/kaggle/working/` "
       "- download from the right sidebar **Output** tab, or after Save & Run All "
       "from the version's Output."),
    code("import glob, os\n",
         "for f in sorted(glob.glob('recover_ne*.csv') + glob.glob('pop_sweep_ne*.csv')):\n",
         "    print(f, f'{os.path.getsize(f)//1024} KB')"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

with open("gate_2_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote gate_2_kaggle.ipynb  (embedded:", ", ".join(SRC) + ")")
