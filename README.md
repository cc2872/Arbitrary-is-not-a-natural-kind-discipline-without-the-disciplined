# norm-extinction-review

Anonymous review snapshot: code and data for "The half-life of a social norm:
decay after enforcement removal in artificial agents" (author identity
withheld for peer review).

## Contents

- `berryworld.py`, `berryworld_jax.py` environment: NumPy reference
  implementation and JAX port.
- `diff_jax_oracle.py` bit-exact diff of the JAX port against the NumPy
  oracle.
- `train_jax.py`, `run_sweep.py`, `rppo.py`, `extinct.py` training,
  parameter-sweep, and enforcement-removal scripts.
- `make_*.py` one script per figure/table, reading run CSVs and writing
  the corresponding output.
- `probe_analysis.py`, `probe_arm3.py`, `vc_analysis.py`, `probe_arm3_*.json`
  probe analyses (coordination, violator-cost, storage).
- `*.ipynb` (root and `notebooks/`) GPU training/sweep notebooks, one per
  run condition.
- `*.csv`, `*.csv.config.json` logged run data and the run configuration
  used to produce each log.
- `fig*.jpg` / `fig*.png` / `fig*.pdf`, `F4_*`, `F5_*`, `new figures/`,
  `table1_anchor_space.*` generated figures and tables.
- `figstyle.py`, `pnas_style.py` shared plotting style.
- `PROBE_PREREGISTRATION.md`, `project_brief.md`, `paper_fills.md`,
  `paper2_summary.py`, `paper2_taxonomy_scaffold.md`, `figure_captions.md`
  dated working notes, including the prespecified decision rules
  referenced in the paper.
- `PROGRESS.md`–`PROGRESS5.md` dated progress logs, AI-written for
  convenience; large-scale data statistics/analysis were also performed by
  AI and checked by a human.
- `Pod_session_1/` separate training workspace (environment copy, training
  script, sweep logs).
- `norm_paper.bib` bibliography.

## License

CC BY 4.0, see `LICENSE`.
