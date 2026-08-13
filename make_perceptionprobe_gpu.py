"""make_perceptionprobe_gpu.py -- notebooks for the perception probe that adjudicates
reading (c) (reconstruction) on the violator-gated persistence null.

Design problem (dictates everything): remove PERCEPTION of enforcement while preserving
its OCCURRENCE. The env has NO zap-beam plane -- enforcement is perceived only through
MARK (violator-status) planes: self_mark (own) + mk world planes (others'). So the probe
masks marks, decomposed into two nested arms run in order:

  M1  self-only  -- zero the agent's OWN self_mark (mask_self_after). Surge-safe BY
                    CONSTRUCTION: enforcers read OTHERS' mk planes, untouched -> the
                    568-zap violator surge cannot degrade. Primary.
  M2  self+world -- additionally zero the mk world planes (unmark_after masks both =
                    M1 U M2). CAN break enforcer targeting -> mandatory surge check.
                    Run only if M1 holds.

Controls:
  PLACEBO -- apply M1's self-mask to the ENFORCER-gated arm (decay driven by incentive
             removal, not perception). Must be behaviorally INERT (pairs vs the existing
             extinct3_enforceronly.csv, same 10 keys). Proves the mask op is a clean scalpel.
  SURGE   -- M2 only: assert ghost zap-rate stays ~500+ (guards occurrence-vs-perception).

Verification already banked (verify_mask.py + diff_jax_oracle): mask-off oracle-diff
0.00e+00; mask_self=True localized to the self-mark feature cols only (world planes bit-
identical); self_cue_on audit column flips 1->0 at the switch. Install phase is unmasked,
so masked and unmasked runs share keys and are bit-identical pre-switch -> probe_analysis
asserts that per seed before pairing.

SCOPE: the mask touches PERCEPTION only, not the reward landscape. A clean HOLD kills (c)
but leaves (a) flat-gradient OPEN (mark-cue mask + standing-stock re-run). If M1/M2 CONFIRM,
that is the expressive-function result (costless-but-perceived enforcement sustains order)
-- report the dissociation and PARK the full story for Paper 2; do not rewrite Paper 1.

Regenerate: python make_perceptionprobe_gpu.py
"""
import json, base64, os
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py", "extinct.py"]
OUTDIR = "notebooks"; os.makedirs(OUTDIR, exist_ok=True)

SETUP = [
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
    "    raise SystemExit('*** Installed. Kernel > Restart, then Run All. ***')\n",
    "import jax\n",
    "print('jax', jax.__version__, '| devices:', jax.devices())\n",
]

# violator-gated env (ghost_keeps_bonus=True): in ghost, enforce gated -> penalty+timeout
# off, enforcer bonus STILL paid -> the 568-zap surge / persistence arm.
ENV_KEEP = [
    "ENV = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
    "                    zap_removal_steps=25, bonus_requires_mark=True,\n",
    "                    c_zapped=2.0, n_berry_types=3, grid=22)\n",
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


def build(nb_name, headline, run_lines, extra_src=("probe_analysis.py",)):
    cells = [
        md(*headline),
        md("## 0. Setup (pure Python; leaves the pod's jax untouched)"),
        code(*SETUP),
        md("## 1. Write sources (base64 -> files; embeds the mask_self machinery + probe_analysis)"),
        _src_cell(list(SRC) + list(extra_src)),
        md("## 2. Run  (expect `>>> RUN START`, per-chunk lines, `>>> RUN DONE`, then the paired verdict)"),
        code(
            "print('>>> RUN START', flush=True)\n",
            "import run_sweep as R, extinct\n",
            *ENV_KEEP,
            *run_lines,
            "\n", "print('>>> RUN DONE', flush=True)\n",
        ),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(OUTDIR, nb_name), "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.join(OUTDIR, nb_name))


# ---- M1: self-perception mask (PRIMARY, surge-preserved by construction) ----
build("probe_m1_gpu.ipynb",
      ["# Perception probe M1 -- self-mask (primary)\n", "\n",
       "Violator-gated ghost (penalty+timeout off, enforcer bonus paid -> zap surge, "
       "berry-1 persistence). At the switch, zero ONLY the agent's own `self_mark` "
       "feature (`mask_self_after=1000`), leaving the world mark planes intact so "
       "enforcer targeting -- and the surge -- are preserved BY CONSTRUCTION.\n", "\n",
       "**Q:** does the disciplined agent's perception of being a marked violator keep it "
       "compliant, when enforcement is otherwise costless? HOLDS (masked opp1 stays ~0.068) "
       "-> (c) dead on the self channel. CONFIRMS (opp1 rises toward 0.35-0.47) -> perceived "
       "enforcement was the whole rule. 10 seeds; unmasked control runs first at the SAME "
       "10 keys (CRN-paired). Reads the silly berry (1,) opportunity-controlled."],
      ["hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
       "# unmasked violator-gated control (10 seeds; same keys as the masked arm)\n",
       "extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=10,\n",
       "                       out_csv='extinct3_m1_unmasked.csv')\n",
       "# M1: self-mark masked from the switch (world mk planes intact -> surge safe)\n",
       "extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=10,\n",
       "                       mask_self_after=1000, out_csv='extinct3_m1_masked.csv')\n",
       "import probe_analysis as P\n",
       "P.report('extinct3_m1_masked.csv', 'extinct3_m1_unmasked.csv',\n",
       "         label='M1 (self-mask, violator-gated)')"])

# ---- M2: ambient mask (COMPANION, run only if M1 holds) + surge check ----
build("probe_m2_gpu.ipynb",
      ["# Perception probe M2 -- self + world (ambient) mask  [run only if M1 HOLDS]\n", "\n",
       "Additionally zero the world mark planes (`unmark_after=1000` masks self AND world "
       "= M1 U M2) -- perception of OTHERS' violator status, the ambient/ritual channel. "
       "This CAN break enforcer targeting, so the run carries a **mandatory surge check**: "
       "ghost zap-rate must stay ~500+ (else occurrence was confounded with perception and "
       "you fall back to M1). Pairs vs the M1 unmasked control (same 10 keys).\n", "\n",
       "**Note (correction to the spec):** the env has no zap-beam plane; 'ambient' here is "
       "others' MARK planes, the only third-person enforcement-perception channel."],
      ["hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
       "import os\n",
       "# reuse M1's unmasked control if present (same 10 keys); else run it\n",
       "if not os.path.exists('extinct3_m1_unmasked.csv'):\n",
       "    extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=10,\n",
       "                           out_csv='extinct3_m1_unmasked.csv')\n",
       "# M2: self + world marks masked from the switch\n",
       "extinct.run_extinction([(1,)], ENV, hp, n_install=1000, n_seeds=10,\n",
       "                       unmark_after=1000, out_csv='extinct3_m2_masked.csv')\n",
       "import probe_analysis as P\n",
       "P.report('extinct3_m2_masked.csv', 'extinct3_m1_unmasked.csv',\n",
       "         label='M2 (self+world, violator-gated)', surge_check=True)"])

# ---- PLACEBO: identical self-mask on the enforcer-gated arm (must be INERT) ----
build("probe_placebo_gpu.ipynb",
      ["# Perception probe -- PLACEBO (self-mask on the enforcer-gated arm)\n", "\n",
       "Apply M1's self-mask to the enforcer-gated arm, where the silly decay is driven by "
       "ENFORCER-INCENTIVE removal (bonus gated at 1000, `enforce` stays on), not by "
       "perception. The mask must be behaviorally INERT: masked decay = unmasked decay. If "
       "so, the mask is a clean scalpel and any violator-arm collapse is removed perception, "
       "not the op. Pairs vs the existing `extinct3_enforceronly.csv` (same 10 keys).\n", "\n",
       "10 seeds. (self_mark is NOT near-empty here -- agents forage and get marked, "
       "prevalence ~0.55 -- so this is a strong inertness test, not a no-op.)"],
      ["hp = dict(R.FAITHFUL_HP); hp['updates']=1600\n",
       "import os\n",
       "# unmasked enforcer-gated control = extinct3_enforceronly.csv (same 10 keys). Run if absent.\n",
       "if not os.path.exists('extinct3_enforceronly.csv'):\n",
       "    extinct.run_extinction([(1,)], ENV, hp, n_install=1600, n_seeds=10,\n",
       "                           gate_bonus_after=1000, out_csv='extinct3_enforceronly.csv')\n",
       "# placebo: enforcer-gated + M1 self-mask at the switch\n",
       "extinct.run_extinction([(1,)], ENV, hp, n_install=1600, n_seeds=10,\n",
       "                       gate_bonus_after=1000, mask_self_after=1000,\n",
       "                       out_csv='extinct3_placebo_m1.csv')\n",
       "import probe_analysis as P\n",
       "P.report('extinct3_placebo_m1.csv', 'extinct3_enforceronly.csv',\n",
       "         label='PLACEBO M1 (enforcer arm; decay must be UNCHANGED)', placebo=True)"])

print("\nRun order: probe_m1_gpu -> (if HOLDS) probe_m2_gpu ; probe_placebo_gpu anytime.")
