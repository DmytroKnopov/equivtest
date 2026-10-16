# Changelog

## v0.2.0 (2026-10-16)

Version of the code accompanying the manuscript submitted to *Machine Learning*.

- `equivtest.power`: `tost_power_exact`, the exact power of the pooled-variance TOST (the one-dimensional integral of Shieh 2016), and `seeds_needed(..., exact=True)`; the shifted-t heuristic is unchanged and remains the default.
- `equivtest.accuracy`: per-run accuracy on a shared test set and its two-level summaries (binomial item-level variance, between-run correlation of correctness, item agreement).
- Second case study on two public collections of independently trained runs: `experiments/case2_modelzoo.py` (CIFAR-10 Model Zoos), `experiments/case2_multiberts.py` (MultiBERTs on RTE, SST-2 and MNLI), `experiments/case2_paired.py`, `experiments/case2_twostage.py`, shared routines in `experiments/case2_common.py`, the vendored model definition in `experiments/modelzoo_defs.py`; results in `results/case2_*.json`, `.md` and per-run `.csv`; Figure 4 in `figures/real_seed_law.*`.
- `experiments/power_exact_check.py` and `results/power_exact_check.*`: at every cell of the power grid and for every design table of the article, the shifted-t heuristic next to the exact pooled power, the Welch-TOST power on normal run means (seeded Monte Carlo) and the realised rate of the two-level simulation.
- `experiments/build_figures.py`: all figures redrawn at the journal's printed width; `experiments/sim_power_robust.py` updated accordingly.
- Tests: `tests/test_accuracy.py`; five exact-power tests in `tests/test_power.py`.
- `CITATION.cff` added; downloaded inputs of the second case study and smoke-run outputs ignored by git.

## v0.1.0 (2026-08-31)

First public version: the `equivtest` package (`auroc`, `dgp`, `tost`, `hierboot`, `power`), the power-validation, robustness and coverage simulation grids, the first case study with its per-item scores, and the regression tests against the published case-study values.
