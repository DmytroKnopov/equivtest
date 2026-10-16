# equivtest

Statistical toolkit for certifying practical equivalence of two training
methods by AUROC: a seed-level Welch-TOST, a two-level (seed × image)
hierarchical bootstrap, and a power analysis answering how many training
runs such a certification requires.

Companion code and data for the article *Equivalence testing and run-count
design for stochastic machine-learning experiments* (D. Knopov and N. Shvai,
2026, submitted). Every number in the article is reproducible from this
repository.

## Installation

```
pip install -e .
```

Requires Python ≥ 3.10, `numpy` and `scipy`; the experiment scripts also
use `matplotlib` for figures and `pytest` for the test suite. The two
scripts of the second case study additionally need `requests`, `torch` and
`torchvision`.

## Quick start

```python
import numpy as np
from equivtest.tost import welch_tost
from equivtest.power import seeds_needed

# per-run AUROCs of two methods (one value per training run)
auroc_a = np.array([0.7003, 0.7155])
auroc_b = np.array([0.6998, 0.7047])

res = welch_tost(auroc_a, auroc_b, margin=0.01, alpha=0.05)
print(res.p_tost, res.ci90, res.equivalent)

# runs per method needed to certify |Delta| < 0.01 with 80% power,
# given a pilot estimate of the per-run standard deviation
print(seeds_needed(margin=0.01, power=0.8, sigma_seed=0.008, v_bar=0.0))
```

## Layout

- `equivtest/` — the package: AUROC and its DeLong variance (`auroc`),
  per-run accuracy and its two-level summaries (`accuracy`), Welch-TOST
  (`tost`), hierarchical bootstrap (`hierboot`), power theory (`power`), and
  the binormal two-level data generator (`dgp`).
- `tests/` — regression tests against the published case-study numbers plus
  synthetic sanity checks. Two tests are slow (a bit-exact replay of the
  published bootstrap stream and a coverage study, ~15–20 min together);
  the rest finish in seconds.
- `experiments/` — the simulation studies and the two case studies of the
  article.
- `results/` — the exact JSON outputs used in the article; the second case
  study adds Markdown renderings and per-run CSV tables.
- `data/` — per-item OOD scores of the four case-study training runs and the
  published summary table the regression tests check against. The scripts of
  the second case study download their inputs into `data/modelzoo/`,
  `data/multiberts/` and `data/derived/`, which are not tracked.

## Reproducing the article

| Article object | Command | Time |
|---|---|---|
| Case study (Tables 1–2, retrospective power, n* = 12) | `python experiments/case_study.py` | ~5 min (`--fast` skips the simulations) |
| Power validation grid (270 cells, both boundaries of the null) | `python experiments/sim_power.py` | ~55 min on 12 cores |
| Non-Gaussian robustness grid (96 cells) | `python experiments/sim_power_robust.py` | ~13 min |
| Exact pooled-model TOST power (Shieh 2016) against the shifted-t heuristic, over the grid and every run count of the article | `python experiments/power_exact_check.py` | seconds (no simulation; reads the result files above) |
| Coverage map (31 cells) | `python experiments/sim_coverage_ext.py` | ~3.5 h |
| Six-cell coverage probe (superseded by the map) | `python experiments/sim_coverage.py` | ~40 min |
| Second case study, CIFAR-10 Model Zoos | `python experiments/case2_modelzoo.py` | 2.8 GB download, then a few minutes on a GPU (`--fast`: 60 runs, 200 replications; `--cpu`: CPU inference) |
| Second case study, MultiBERTs | `python experiments/case2_multiberts.py` | 230 MB download, then a few minutes (`--fast`: 200 replications) |
| Figures 1–4 | `python experiments/build_figures.py` (`fig4` alone rebuilds Figure 4) | seconds |

All root seeds are fixed inside the scripts; re-running a study reproduces
the shipped JSON in `results/` up to floating-point identity. The two tables
of the second case study also carry a provenance block (date, runtime,
device, software versions) that differs between machines.

## Data

`data/ood_scores/` holds the raw per-item scores of the four training runs
of the case study — a joint energy-based model on CIFAR-10, trained with the
SGLD and predictor–corrector samplers, two seeds each — on the CIFAR-10 test
set (in-distribution) and five OOD sets (SVHN, CIFAR-100, DTD, LSUN-R, iSUN).
`data/hier_bootstrap_tost.json` is the summary table published with the case
study; `tests/` reproduce its numbers bit-exactly from the raw scores.

## Second case study

The second case study of the article applies the same protocol to
classification accuracy on two public collections of independently trained
runs. Neither collection is shipped here; the two scripts download what they
need (about 3 GB in total) on their first run and cache it under `data/`.

- **Model Zoos** (Schürholt et al., 2022; Zenodo record 6620869, CC BY 4.0).
  Two archives of the CIFAR-10 CNN-s zoos: `cifar_small_seed.zip` (0.9 GB;
  one configuration trained with seeds 1–1000) and `cifar_small_hyp_fix.zip`
  (1.9 GB; 128 configurations × 10 seeds). `experiments/case2_modelzoo.py`
  downloads and checksums the archives, extracts the last checkpoint of every
  trial and recomputes per-image correctness on the 10,000 CIFAR-10 test
  images (fetched by `torchvision` into `data/`) with the authors' model
  definition, vendored in `experiments/modelzoo_defs.py` with its provenance
  in the file header. Inference runs on the GPU when one is available
  (`--cpu` forces the CPU); `--fast` evaluates 60 seed-zoo runs with 200
  replications instead of 1000 runs with 5000.
- **MultiBERTs** (Sellam et al., 2022). Per-example dev-set predictions of the
  fine-tuning runs released by the authors on public Google Cloud Storage
  (`storage.googleapis.com/multiberts/public/example-predictions/GLUE/`,
  620 prediction files per task); the authors state no licence for these
  files. `experiments/case2_multiberts.py` fetches the RTE, SST-2 and MNLI
  files (about 230 MB) and scores the argmax of the released probabilities
  against the released labels; `--fast` uses 200 replications instead of
  2000.

Both scripts run from the repository root. On Windows, set
`KMP_DUPLICATE_LIB_OK=TRUE` if `torch` complains about a duplicate OpenMP
runtime.

What the result tables contain:

- `results/case2_modelzoo.json` / `.md` — verification of the recomputed
  accuracies against the shipped ones; the two-level structure of the 1000
  seed-zoo accuracies (moments, binomial item-level variance and its ratio to
  s^2, between-run correlation of correctness, item agreement, design table
  n*(margin, power)); the seed law on real runs (rate of declaring
  equivalence at Delta = 0 and realised size at Delta = +-margin for disjoint
  arms of n runs, margins 0.01 and 0.0173, against the formula); and two
  genuine comparisons in the hyperparameter zoo (Adam vs SGD, GELU vs tanh:
  full-data TOST, certification rate at small n, pilot-then-design).
  `results/case2_runs_modelzoo.csv` lists every evaluated run with its
  shipped and recomputed accuracy.
- `results/case2_multiberts.json` / `.md` — per task: the summary of the 25
  pretraining seeds of one configuration (2M steps, lr 2e-5, fine-tuning
  seed 0), the split of the accuracy variance into pretraining and
  fine-tuning components, the comparison lr 2e-5 vs 5e-5 (25 vs 25 runs),
  the comparison 1M vs 2M pretraining steps (5 vs 5, unpaired and paired
  TOST), and a null-true check on disjoint arms.
  `results/case2_runs_multiberts.csv` gives the accuracy of each of the 1860
  runs.
- `figures/real_seed_law.pdf` (Figure 4) is drawn from the two JSON files by
  `python experiments/build_figures.py fig4`.
- `results/power_exact_check.json` / `.md` — at every cell of
  `sim_power.json`, the shifted-t heuristic next to three reference values:
  the exact pooled-model TOST power (`equivtest.power.tost_power_exact`, the
  one-dimensional integral of Shieh 2016), the Welch-TOST power on normal run
  means with a common variance (Monte Carlo, 200 000 replications per cell,
  fixed root seed) and the realised rate of the two-level simulation, so that
  the approximation error, the pooled-to-Welch gap and the departure of the
  AUROC process from normal run means are separated; summaries by n, inside
  the margin, at its boundary and outside it. Also the exact run counts next
  to the heuristic ones for every design table of the article (first case
  study, single-configuration collections, two-stage margins, planning
  column of the paired comparisons).

`equivtest.accuracy` holds the accuracy-side helpers: `accuracy(correct)`
(mean of a 0/1 vector), `binomial_variance(p, m)` (p(1 - p)/m, the
item-level variance of a run's accuracy given the run),
`between_run_correlation(correct)` (mean pairwise Pearson correlation of the
rows of an R × m correctness matrix, computed without forming the R × R
matrix) and `item_agreement(correct)` (shares of items always, never and
sometimes correct across the runs).

## Licence

MIT.
