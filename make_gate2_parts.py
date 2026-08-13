"""Generate the Gate-2 pipeline as THREE self-contained, hand-off-by-CSV Colab
notebooks, so each stage fits inside one Colab session and you carry state
forward by downloading a CSV and uploading it into the next part:

    gate2_part1_recover.ipynb  (GPU)  -> recover_ne<NE>.csv   (one per num_envs)
    gate2_part2_pilot.ipynb    (GPU)  <- upload recover_ne*.csv  -> pop_sweep_ne<BEST>.csv
    gate2_part3_verdict.ipynb  (CPU)  <- upload pop_sweep_ne<BEST>.csv  -> printed verdict

Each part embeds the current .py sources via %%writefile (nothing to upload but
the CSVs; no drift). Regenerate after any .py edit:  python make_gate2_parts.py

Why split: sequential-seed runs trade memory for wall-clock, so the full
num_envs recovery + pilot can exceed a single Colab session. Part 1's NE_LIST is
shrinkable -- run {64,128} in one session, {256,512} in the next; every
num_envs writes its own recover_ne<NE>.csv, and Part 2 picks across all of them.
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


ENV_CELL = code(
    "import importlib, subprocess, sys\n",
    "missing = [m for m in ('flax', 'optax') if importlib.util.find_spec(m) is None]\n",
    "if missing:\n",
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=False)\n",
    "import jax, flax, optax, numpy as np\n",
    "print('jax', jax.__version__, ' flax', flax.__version__, ' optax', optax.__version__)\n",
    "devs = jax.devices(); print('devices:', devs)\n",
    "print('OK - GPU ready.' if any(d.platform == 'gpu' for d in devs)\n",
    "      else '[!] No GPU. Runtime -> Change runtime type -> GPU, then Run all.')")

WRITE_CELLS = [md("## Write the source files (self-contained; seeds run "
                  "sequentially -> ~5x less VRAM)")] + [writefile_cell(p) for p in SRC]


def dump(path, cells, gpu=True):
    meta = {"colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
    nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


# ============================================================ PART 1 : RECOVER
part1 = [
    md("# Gate 2 - Part 1/3: recover the koster num_envs (GPU)\n",
       "\n",
       "Runs the exact koster grid at N=12 (D=100, bonus=8.75, ep=300, "
       "removal=25, **pending hidden**, 5 seeds, 800 updates), swept over "
       "`num_envs`. Seeds run **sequentially** (memory-safe). Each `num_envs` "
       "writes its own `recover_ne<NE>.csv`.\n",
       "\n",
       "The koster signature is `none` eat0 ~91 (never learns to avoid poison) "
       "and `none - (0,1)` ~45. **Download every `recover_ne*.csv` at the end** - "
       "Part 2 reads them to pick the recovered config.\n",
       "\n",
       "**Session budget:** if the full list is too long for one session, shrink "
       "`NE_LIST` (e.g. `[64, 128]` now, `[256, 512]` in a second run) - the CSVs "
       "accumulate across sessions."),
    md("## 0. Environment"), ENV_CELL,
    *WRITE_CELLS,
    md("## 2. Recover: sweep num_envs (edit NE_LIST to fit your session)"),
    code("import run_sweep as R, numpy as np\n",
         "\n",
         "NE_LIST = [64, 128, 256, 512]     # <-- shrink to fit a session if needed\n",
         "KOSTER_ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75,\n",
         "                           episode_len=300, zap_removal_steps=25)  # pending hidden\n",
         "\n",
         "print('num_envs recovery (N=12, koster grid, sequential seeds)\\n')\n",
         "print(f\"{'num_envs':>8} | {'none':>6} {'(0,)':>6} {'(0,1)':>6} | \"\n",
         "      f\"{'none-(0,1)':>11} | recovers?\")\n",
         "print('-' * 60)\n",
         "for ne in NE_LIST:\n",
         "    hp = dict(R.FAITHFUL_HP); hp['updates'] = 800; hp['num_envs'] = ne\n",
         "    rows = R.run_sweep([12], [(), (0,), (0, 1)], n_seeds=5, hp=hp,\n",
         "                       env_list=[KOSTER_ENV], out_csv=f'recover_ne{ne}.csv')\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(0.1 * (umax + 1))\n",
         "    e = {c: np.mean([float(r['eat0']) for r in rows\n",
         "                     if r['condition'] == c and int(r['update']) >= cut])\n",
         "         for c in ('none', '0', '01')}\n",
         "    diss = e['none'] - e['01']; rec = e['none'] > 70 and diss > 25\n",
         "    print(f\"{ne:>8} | {e['none']:>6.1f} {e['0']:>6.1f} {e['01']:>6.1f} | \"\n",
         "          f\"{diss:>11.1f} | {'YES' if rec else 'no'}\")\n",
         "print('\\nDownload the recover_ne*.csv below and upload them into Part 2.')"),
    md("## 3. Download the recovery CSVs (-> upload into Part 2)"),
    code("from google.colab import files\n",
         "import glob\n",
         "for fpath in sorted(glob.glob('recover_ne*.csv')):\n",
         "    print('downloading', fpath); files.download(fpath)"),
]

# ============================================================ PART 2 : PILOT
part2 = [
    md("# Gate 2 - Part 2/3: population pilot at the recovered num_envs (GPU)\n",
       "\n",
       "**Upload the `recover_ne*.csv` files from Part 1** in section 2. This part "
       "auto-picks `BEST_NE` = the smallest `num_envs` that reproduced koster "
       "(`none` ~91, `none-(0,1)` ~25+), then runs the population pilot "
       "`N in {6,8,10,12}` x 3 conditions x 5 seeds at that config, writing "
       "`pop_sweep_ne<BEST_NE>.csv`.\n",
       "\n",
       "If Part 1 recovered nothing, this part stops - go back and try a higher "
       "`num_envs` (1024) or a fresh runtime (bistability check). You can also set "
       "`BEST_NE` by hand in section 3 to override the auto-pick."),
    md("## 0. Environment"), ENV_CELL,
    *WRITE_CELLS,
    md("## 2. Upload the recover_ne*.csv from Part 1"),
    code("from google.colab import files\n",
         "print('Select the recover_ne*.csv files you downloaded from Part 1...')\n",
         "up = files.upload()          # writes them into the working dir\n",
         "print('uploaded:', list(up))"),
    md("## 3. Pick BEST_NE from the recovery CSVs (or set it manually)"),
    code("import csv, glob, numpy as np\n",
         "\n",
         "BEST_NE = None       # <-- set an int here to override the auto-pick\n",
         "\n",
         "def _none_regime(path):\n",
         "    rows = list(csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(0.1 * (umax + 1))\n",
         "    e = {c: np.mean([float(r['eat0']) for r in rows\n",
         "                     if r['condition'] == c and int(r['update']) >= cut])\n",
         "         for c in ('none', '0', '01')}\n",
         "    return e['none'], e['none'] - e['01']\n",
         "\n",
         "if BEST_NE is None:\n",
         "    cands = []\n",
         "    for path in sorted(glob.glob('recover_ne*.csv')):\n",
         "        ne = int(path.split('ne')[1].split('.')[0])\n",
         "        none, diss = _none_regime(path)\n",
         "        rec = none > 70 and diss > 25\n",
         "        print(f'{path}: none={none:.1f}  none-(0,1)={diss:.1f}  recovers={rec}')\n",
         "        if rec: cands.append(ne)\n",
         "    BEST_NE = min(cands) if cands else None\n",
         "\n",
         "print('\\nBEST_NE =', BEST_NE)\n",
         "if BEST_NE is None:\n",
         "    print('>>> No num_envs recovered koster. STOP: try 1024 / a fresh runtime.')"),
    md("## 4. Run the population pilot at BEST_NE"),
    code("import pop_sweep as P\n",
         "assert BEST_NE is not None, 'No recovered num_envs -- resolve Part 1 first.'\n",
         "print(f'Population pilot at num_envs={BEST_NE}, pending hidden, N in {P.N_GRID}\\n')\n",
         "P.run(num_envs=BEST_NE)\n",
         "P.analyze(f'pop_sweep_ne{BEST_NE}.csv')"),
    md("## 5. Download the pilot CSV (-> upload into Part 3)"),
    code("from google.colab import files\n",
         "files.download(f'pop_sweep_ne{BEST_NE}.csv')"),
]

# ============================================================ PART 3 : VERDICT
part3 = [
    md("# Gate 2 - Part 3/3: read the verdict (CPU - no GPU needed)\n",
       "\n",
       "**Upload the `pop_sweep_ne<BEST_NE>.csv` from Part 2** in section 1. Prints "
       "the paired berry-1 norm signal `d1 = eat1[(0,)] - eat1[(0,1)]` per N "
       "(SURVIVES at margin>=2) and the compliance regime check. The lowest "
       "surviving N is the population the extinction experiment runs at.\n",
       "\n",
       "Pure analysis - runs on a CPU runtime. No source files or GPU required."),
    md("## 1. Upload the pilot CSV from Part 2"),
    code("from google.colab import files\n",
         "up = files.upload()\n",
         "CSV = list(up)[0]; print('analyzing', CSV)"),
    md("## 2. Verdict"),
    code("import csv as _csv, numpy as np\n",
         "\n",
         "def tail_cells(path, tail_frac=0.1):\n",
         "    rows = list(_csv.DictReader(open(path)))\n",
         "    umax = max(int(r['update']) for r in rows); cut = umax - int(tail_frac * (umax + 1))\n",
         "    acc = {}\n",
         "    for r in rows:\n",
         "        if int(r['update']) < cut: continue\n",
         "        k = (int(r['N']), r['condition'], int(r['seed']))\n",
         "        acc.setdefault(k, []).append((float(r['eat0']), float(r['eat1'])))\n",
         "    cells = {}\n",
         "    for (N, cond, s), vals in acc.items():\n",
         "        cells.setdefault(N, {}).setdefault(cond, {})[s] = np.array(vals).mean(0)\n",
         "    return cells\n",
         "\n",
         "def paired(a, b):\n",
         "    seeds = sorted(set(a) & set(b)); d = np.array([a[s] - b[s] for s in seeds])\n",
         "    if len(d) < 2: return (float(d.mean()) if len(d) else float('nan')), float('nan'), len(d)\n",
         "    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)\n",
         "\n",
         "cells = tail_cells(CSV)\n",
         "print('berry-1 norm signal d1 = eat1[(0,)] - eat1[(0,1)], paired; SURVIVES if margin>=2\\n')\n",
         "print(f\"{'N':>3} | {'eat1 (0,)':>10} {'eat1 (0,1)':>11} | {'d1':>7} {'SEM':>6} {'margin':>7} | verdict\")\n",
         "print('-' * 72)\n",
         "for N in sorted(cells):\n",
         "    c = cells[N]\n",
         "    if '0' not in c or '01' not in c:\n",
         "        print(f'{N:>3} | incomplete'); continue\n",
         "    a = {s: v[1] for s, v in c['0'].items()}; b = {s: v[1] for s, v in c['01'].items()}\n",
         "    m, sem, n = paired(a, b)\n",
         "    margin = m / sem if sem and np.isfinite(sem) and sem > 0 else float('nan')\n",
         "    surv = 'SURVIVES' if (np.isfinite(margin) and margin >= 2) else 'weak/none'\n",
         "    print(f'{N:>3} | {np.mean(list(a.values())):>10.1f} {np.mean(list(b.values())):>11.1f} | '\n",
         "          f'{m:>7.2f} {sem:>6.2f} {margin:>7.2f} | {surv}')\n",
         "print('\\ncompliance regime check (koster = none ~90 >> (0,1) ~45):')\n",
         "print(f\"{'N':>3} | {'none':>7} {'(0,)':>7} {'(0,1)':>7}\")\n",
         "for N in sorted(cells):\n",
         "    c = cells[N]\n",
         "    g = lambda cc: np.mean([v[0] for v in c[cc].values()]) if cc in c else float('nan')\n",
         "    print(f'{N:>3} | {g(\"none\"):>7.1f} {g(\"0\"):>7.1f} {g(\"01\"):>7.1f}')"),
]

dump("gate2_part1_recover.ipynb", part1, gpu=True)
dump("gate2_part2_pilot.ipynb", part2, gpu=True)
dump("gate2_part3_verdict.ipynb", part3, gpu=False)
print("done - 3 parts, hand off by CSV (part1 -> part2 -> part3)")
