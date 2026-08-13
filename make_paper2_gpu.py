"""Paper 2 (norm-grounding taxonomy) GPU notebooks -> ./notebooks/. Same robust
3-cell / no-shell / base64-source format as make_confirm_runs.py; embeds CURRENT
sources (coordination mechanic + conformity polarity + hazard/flatten flags, all
oracle-diffed 0.00e+00). Regenerate: python make_paper2_gpu.py

Family (install 0-999 / ghost 1000-1599, ghost_keeps_bonus=False = valid oversight removal):
  ENVIRONMENTAL own-knockout (pairs vs paper-1 poison-ghost):
    poison_hazardoff        (5s)  ghost + hazard_off_after=1000  -> poison DECAYS
    poison_hazardoff_10     (10s) confirmatory power
  COORDINATION row (convergence norm; convergent_berry=1 + conformity_berry=1):
    coord_cleanghost        (5s)  ghost only            -> convergence PERSISTS (feasibility fuse in its install phase)
    coord_flat              (5s)  ghost + flatten@1000  -> convergence DECAYS (grounding knockout)
    coord_cleanghost_10     (10s) confirmatory
    coord_flat_10           (10s) confirmatory
Coordination readout uses extinct.analyze(..., converge={1}): the norm is opp_C HIGH,
so decay = opp_C FALLS (sign-flipped so >0 still = decay). coord_k=0.15 is a calibration
GUESS -- coord_cleanghost's install phase is the cheap fuse: if convergence doesn't
install (opp1 not driven high, selectivity<1), raise coord_k / r_zap_bonus and rerun one.
"""
import json, base64, os
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py", "extinct.py"]
OUTDIR = "notebooks"; os.makedirs(OUTDIR, exist_ok=True)

SETUP = [   # ensure GPU-bound jax; install jax[cuda12] + restart if not (see make_confirm_runs)
    "import os, sys, subprocess\n",
    "os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')\n",
    "def _pip(*pkgs): subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])\n",
    "try:\n",
    "    import jax, flax, optax\n",
    "    on_gpu = jax.devices()[0].platform == 'gpu'\n",
    "except Exception:\n",
    "    on_gpu = False\n",
    "if not on_gpu:\n",
    "    print('No GPU-bound jax -> installing jax[cuda12], flax, optax ...', flush=True)\n",
    "    _pip('jax[cuda12]', 'flax', 'optax')\n",
    "    raise SystemExit('*** Installed. Now: Kernel > Restart, then Run All. '\n",
    "                     '(If STILL CpuDevice after restart, this pod has NO GPU attached.) ***')\n",
    "import jax\n",
    "print('jax', jax.__version__, '| devices:', jax.devices())\n",
]


def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": list(l)}
def code(*l): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(l)}


def _src_cell(paths):
    lines = ["import base64, pathlib\n", "_SRC = {\n"]
    for p in paths:
        with open(p, "rb") as f:
            lines.append(f"  {p!r}: {base64.b64encode(f.read()).decode()!r},\n")
    lines += ["}\n",
              "for _n, _b in _SRC.items(): pathlib.Path(_n).write_bytes(base64.b64decode(_b))\n",
              "print('wrote sources:', list(_SRC))\n"]
    return code(*lines)


def build(nb_name, out_csv, headline, run_lines):
    cells = [
        md(*headline),
        md("## 0. Setup (pure Python; leaves the pod's jax untouched)"),
        code(*SETUP),
        md("## 1. Write sources (base64 -> files; incl. coordination/conformity/hazard flags)"),
        _src_cell(SRC),
        md("## 2. Run  (expect `>>> RUN START`, per-chunk lines, then `>>> RUN DONE`)"),
        code("print('>>> RUN START', flush=True)\n",
             "import run_sweep as R, extinct\n", *run_lines,
             "\n", "print('>>> RUN DONE', flush=True)\n"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(OUTDIR, nb_name), "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.join(OUTDIR, nb_name), "->", out_csv)


ENV_FULL = ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True,\n"
            "                    c_zapped=2.0, n_berry_types=3, grid=22, ghost_keeps_bonus=False)\n")
ENV_COORD = ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
             "                    zap_removal_steps=25, bonus_requires_mark=True,\n"
             "                    c_zapped=2.0, n_berry_types=3, grid=22, ghost_keeps_bonus=False,\n"
             "                    convergent_berry=1, coord_k=2.0, coord_a=1.5, conformity_berry=1)\n")


# ---- ENVIRONMENTAL own-knockout ----
for tag, seeds in [("poison_hazardoff", 5), ("poison_hazardoff_10", 10)]:
    build(f"{tag}_gpu.ipynb", f"{tag}.csv",
          [f"# Environmental own-knockout: ghost + hazard-off (poison), {seeds} seeds\n", "\n",
           "Full oversight removal (`ghost_keeps_bonus=False`) PLUS `hazard_off_after=1000` "
           "gating the -4 physical penalty in phase 2. Pairs vs paper-1 poison-ghost "
           "(`extinct3_cleanghost.csv` cond (0,)): ghost-alone PERSISTS, ghost+hazard-off "
           "DECAYS = the environmental-grounding proof."],
          [ENV_FULL, "hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
           f"extinct.run_extinction([(0,)], ENV, hp, n_install=1000, n_seeds={seeds},\n",
           f"                       hazard_off_after=1000, vmap_seeds=5,\n",
           f"                       out_csv='{tag}.csv')\n",
           f"extinct.analyze('{tag}.csv')"])

# ---- COORDINATION row (convergence norm; conformity enforcement) ----
COORD_HEAD = (
    "The coordination norm is a CONVENTION grounded in increasing-returns co-eating "
    "(`convergent_berry=1`, **coord_k=2.0**, coord_a=1.5). Installed under conformity "
    "enforcement (`conformity_berry=1` -> a violator hasn't eaten berry 1 within "
    "mark_steps). Readout uses `converge={1}` so decay = opp_1 FALLING vs the constant-"
    "returns companion (berry 2). FRAMING (A): remove enforcement (cleanghost) -> "
    "convergence PERSISTS at a coordination-sustained floor; flatten the grounding (flat) "
    "-> convergence COLLAPSES. The no-enforcement baseline (coord_baseline) is the control "
    "proving the ground is coordination, not oversight (convergence self-installs there).\n")
for tag, seeds, extra in [
    ("coord_cleanghost", 5, ""),
    ("coord_flat", 5, "                       flatten_returns_after=1000,\n"),
    ("coord_cleanghost_10", 10, ""),
    ("coord_flat_10", 10, "                       flatten_returns_after=1000,\n"),
]:
    knockout = "grounding knockout (flatten returns at 1000) -> convergence DECAYS" if "flat" in tag \
        else "enforcement removal only (ghost) -> convergence PERSISTS (coordination sustains)"
    build(f"{tag}_gpu.ipynb", f"{tag}.csv",
          [f"# Coordination row: {tag} ({seeds} seeds)\n", "\n", f"Phase 2 = {knockout}.\n", "\n", COORD_HEAD],
          [ENV_COORD, "hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
           f"extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds={seeds},\n",
           extra + "                       vmap_seeds=5,\n",
           f"                       out_csv='{tag}.csv')\n",
           f"extinct.analyze('{tag}.csv', converge={{1}})"])


# ---- coord_k CALIBRATION: one coord_k per GPU, each runs the FULL cell (cleanghost +
#      flat). coord_k=0.15 was too weak (convention decayed). These bracket above it; the
#      winner = lowest coord_k where cleanghost PERSISTS (rel_decay~0) AND flat DECAYS. ----
def env_coord(k):
    return ("ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
            "                    zap_removal_steps=25, bonus_requires_mark=True,\n"
            "                    c_zapped=2.0, n_berry_types=3, grid=22, ghost_keeps_bonus=False,\n"
            f"                    convergent_berry=1, coord_k={k}, coord_a=1.5, conformity_berry=1)\n")
# ---- coord NO-ENFORCEMENT baseline: does convergence self-organize from coordination
#      returns ALONE (conformity_berry=None -> no enforcement)? The k=2.0 sel<1 result
#      makes this load-bearing: if convergence emerges WITHOUT enforcement, the row is
#      "coordination self-installs" (not "enforcement installs, coordination sustains").
#      Swept over coord_k so it lines up with the calibration; readout = opp1 vs companion.
_READOUT = [
    "import csv as _csv, statistics as _st\n",
    "def conv_readout(path, C=1, comp=2):\n",
    "    r=[x for x in _csv.DictReader(open(path))]; seeds=sorted(set(int(x['seed']) for x in r))\n",
    "    def opp(s,i):\n",
    "        rs=[x for x in r if int(x['seed'])==s and int(x['update'])>=900]\n",
    "        e=sum(float(x['eat'+str(i)]) for x in rs); c=sum(float(x['enc'+str(i)]) for x in rs)\n",
    "        return e/max(c,1e-9)\n",
    "    o1=_st.mean([opp(s,C) for s in seeds]); o2=_st.mean([opp(s,comp) for s in seeds])\n",
    "    v='SELF-INSTALLS (enforcement NOT needed)' if o1-o2>0.05 else 'no self-install (enforcement needed)'\n",
    "    print(path,'opp%d(convergent)=%.3f  opp%d(companion)=%.3f  pref=%+.3f  -> %s'%(C,o1,comp,o2,o1-o2,v))\n",
]
_base_rl = ["hp = dict(R.FAITHFUL_HP); hp['updates']=1000\n", *_READOUT]
for k, ktag in [(0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0")]:
    _base_rl += [
        "ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n"
        "                    zap_removal_steps=25, bonus_requires_mark=True,\n"
        "                    c_zapped=2.0, n_berry_types=3, grid=22,\n"
        f"                    convergent_berry=1, coord_k={k}, coord_a=1.5)  # conformity_berry=None => NO enforcement\n",
        f"extinct.run_extinction([()], ENV, hp, n_install=1000, n_seeds=5, vmap_seeds=5,\n",
        f"                       out_csv='coord_baseline_k{ktag}.csv')\n",
        f"conv_readout('coord_baseline_k{ktag}.csv')\n"]
build("coord_baseline_gpu.ipynb", "coord_baseline_k*.csv",
      ["# Coordination NO-ENFORCEMENT baseline (does the convention self-install?)\n", "\n",
       "Coordination returns ON (`convergent_berry=1`, coord_k swept 0.5/1.0/2.0) but "
       "`conformity_berry=None` -> NO enforcement, and condition `()` -> nothing marked. "
       "If opp1 (convergent berry) rises clearly above the companion, the convention "
       "self-installs from coordination alone (enforcement redundant, as the k=2.0 sel<1 "
       "result hinted); if opp1 ~ companion, enforcement is genuinely doing the install "
       "work. This decides what the whole coordination ROW claims. 1000 updates, 5 seeds, "
       "run on one GPU (~1.5 hr).\n"],
      _base_rl)


for k, ktag in [(0.5, "0p5"), (0.75, "0p75"), (1.0, "1p0"), (1.5, "1p5"), (2.0, "2p0")]:
    build(f"coord_cal_k{ktag}_gpu.ipynb", f"coord_*_k{ktag}.csv",
          [f"# Coordination coord_k calibration: coord_k={k} (run on its own GPU)\n", "\n",
           f"coord_k=0.15 gave a WEAK convention that DECAYED. This runs the full coordination "
           f"cell at coord_k={k}: cleanghost (enforcement removed -> should PERSIST if the grounding "
           f"is now strong enough) + flat (grounding flattened -> should DECAY). Winner across the "
           f"three GPUs = lowest coord_k where cleanghost persists (rel_decay~0, margin<2) AND flat "
           f"decays (rel_decay>0, margin>2). Also eyeball install-end opp1 vs companion (strong "
           f"convention = opp1 clearly > companion) and selectivity>1.\n", "\n", COORD_HEAD],
          [env_coord(k), "hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
           f"# 1) cleanghost (persistence) at coord_k={k}\n",
           f"extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=5, vmap_seeds=5,\n",
           f"                       out_csv='coord_cleanghost_k{ktag}.csv')\n",
           f"extinct.analyze('coord_cleanghost_k{ktag}.csv', converge={{1}})\n",
           f"# 2) flat (grounding knockout) at coord_k={k}\n",
           f"extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=5,\n",
           f"                       flatten_returns_after=1000, vmap_seeds=5,\n",
           f"                       out_csv='coord_flat_k{ktag}.csv')\n",
           f"extinct.analyze('coord_flat_k{ktag}.csv', converge={{1}})"])
