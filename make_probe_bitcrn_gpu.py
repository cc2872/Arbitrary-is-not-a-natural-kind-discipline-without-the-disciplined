"""make_probe_bitcrn_gpu.py -- ONE consolidated bit-CRN session that closes all three
loose ends from the first probe run in a single codegen:
  * unreliable deltas    -> masked & unmasked share keys AND compilation -> bit-exact CRN
  * pre-reg evaluability -> the margin-based HOLD rule can finally be applied (pairing valid)
  * placebo certification -> placebo paired vs a SAME-SESSION enforcer control (not stale CSV)

Why one notebook: the earlier install bit-identity failures were cross-process /
mixed-batching codegen drift (chaotic amplification through 900 install updates, seeded
by discrete-action sampling). The fix is to run every arm in ONE process, ONE GPU, with
ONE `vmap_seeds` value, so masked and unmasked go through the identical compiled graph and
are bit-identical PRE-switch (probe_analysis asserts this before pairing).

Arms (all n_seeds=10, cond (1,) silly, same PRNGKey(0) split -> matched keys):
  violator-gated (ENV_KEEP):   unmasked | M1 mask_self_after=1000 | M2 unmark_after=1000
  enforcer-gated (n_install=1600, gate_bonus_after=1000):  unmasked | placebo mask_self_after=1000

Each arm writes its own CSV and is guarded by a done() check -> if the session is killed
mid-arm, re-running the notebook skips completed arms (arm-level resume; no driver change).

Regenerate: python make_probe_bitcrn_gpu.py
"""
import json, base64, os
SRC = ["berryworld.py", "berryworld_jax.py", "train_jax.py", "run_sweep.py", "extinct.py"]
OUTDIR = "notebooks"; os.makedirs(OUTDIR, exist_ok=True)

# VS = vmap_seeds for EVERY arm (one value -> one codegen). Set by the CPU bit-CRN test:
#   False (sequential) is guaranteed bit-CRN; True (chunked vmap) only if the test says so.
VS = True    # chunked vmap: proven bit-CRN pre-switch (0.00e+00) same as sequential, but faster on A100

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


def build():
    run = [
        "print('>>> RUN START', flush=True)\n",
        "import os, csv, run_sweep as R, extinct\n",
        f"VS = {VS!r}   # vmap_seeds for EVERY arm -> one codegen (bit-CRN precondition)\n",
        "def done(p):\n",
        "    return os.path.exists(p) and sum(1 for _ in open(p)) > 1   # header + >=1 data row\n",
        "def go(csvname, **kw):\n",
        "    if done(csvname):\n",
        "        print(f'  SKIP {csvname} (already has data)', flush=True); return\n",
        "    extinct.run_extinction([(1,)], kw.pop('env'), hp, n_seeds=10, vmap_seeds=VS,\n",
        "                           out_csv=csvname, **kw)\n",
        "\n",
        "# violator-gated env (ghost_keeps_bonus=True): penalty+timeout off in ghost, bonus paid -> surge\n",
        "ENV_V = R.env_variant(poison_delay=100, r_zap_bonus=8.75, episode_len=300,\n",
        "                      zap_removal_steps=25, bonus_requires_mark=True,\n",
        "                      c_zapped=2.0, n_berry_types=3, grid=22)\n",
        "hp = dict(R.FAITHFUL_HP); hp['updates'] = 1600\n",
        "\n",
        "# --- violator-gated triplet (unmasked control + M1 + M2), matched keys+codegen ---\n",
        "go('extinct3_bitcrn_v_unmasked.csv', env=ENV_V, n_install=1000)\n",
        "go('extinct3_bitcrn_v_m1.csv',       env=ENV_V, n_install=1000, mask_self_after=1000)\n",
        "go('extinct3_bitcrn_v_m2.csv',       env=ENV_V, n_install=1000, unmark_after=1000)\n",
        "\n",
        "# --- enforcer-gated pair (unmasked control + placebo self-mask), matched keys+codegen ---\n",
        "go('extinct3_bitcrn_e_unmasked.csv', env=ENV_V, n_install=1600, gate_bonus_after=1000)\n",
        "go('extinct3_bitcrn_e_placebo.csv',  env=ENV_V, n_install=1600, gate_bonus_after=1000, mask_self_after=1000)\n",
        "print('>>> RUN DONE', flush=True)\n",
        "\n",
        "# --- verdicts: install bit-identity must now PASS (bit-CRN) -> margins evaluable ---\n",
        "import probe_analysis as P\n",
        "print('\\n=========== M1 (self-mask, PRIMARY) ===========')\n",
        "P.report('extinct3_bitcrn_v_m1.csv', 'extinct3_bitcrn_v_unmasked.csv', label='M1 self-mask')\n",
        "print('\\n=========== M2 (self+world, +surge check) ===========')\n",
        "P.report('extinct3_bitcrn_v_m2.csv', 'extinct3_bitcrn_v_unmasked.csv', label='M2 self+world', surge_check=True)\n",
        "print('\\n=========== PLACEBO (enforcer arm; must be INERT) ===========')\n",
        "P.report('extinct3_bitcrn_e_placebo.csv', 'extinct3_bitcrn_e_unmasked.csv', label='PLACEBO', placebo=True)\n",
    ]
    cells = [
        md("# Perception probe -- consolidated bit-CRN session (M1 + M2 + placebo)\n", "\n",
           "One process, one GPU, one `vmap_seeds` -> every arm shares compilation, and each "
           "masked/unmasked pair shares keys -> **bit-exact CRN**. This is what makes the "
           "margin-based HOLD criterion evaluable (the first run's install bit-identity failed, "
           "so the deltas couldn't be read against the pre-registration) and lets the placebo "
           "be certified at +-0.02 against a same-session control.\n", "\n",
           "Verdict rides on the **absolute** masked ghost-end opp1 (persistence ~0.07 vs decay "
           "band 0.35-0.47) -- pairing-free -- but the deltas/margins now become trustworthy. "
           "Arms are guarded by `done()`: a killed session resumes at the first unfinished arm."),
        md("## 0. Setup"),
        code(*SETUP),
        md("## 1. Write sources (embeds mask_self machinery + probe_analysis)"),
        _src_cell(list(SRC) + ["probe_analysis.py"]),
        md("## 2. Run all arms + verdicts  (expect per-arm lines, then three `HOLDS/INERT` reports)"),
        code(*run),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(OUTDIR, "probe_bitcrn_gpu.ipynb"), "w", encoding="utf-8"), indent=1)
    print(f"wrote notebooks/probe_bitcrn_gpu.ipynb  (VS/vmap_seeds={VS})")


if __name__ == "__main__":
    build()
