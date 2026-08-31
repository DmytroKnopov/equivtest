# equivtest

Statistical toolkit for certifying practical equivalence of two training
methods by AUROC: a seed-level Welch-TOST, a two-level (seed × image)
hierarchical bootstrap, and a power analysis answering how many training
runs such a certification requires.

Companion code and data for the article *Equivalence testing and power
analysis for comparing machine-learning methods by AUROC* (D. Knopov, 2026,
submitted). Every number in the article is reproducible from this repository.

## Installation

```
pip install -e .
```

Requires Python ≥ 3.10, `numpy` and `scipy`; the experiment scripts also
use `matplotlib` for figures and `pytest` for the test suite.

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
  Welch-TOST (`tost`), hierarchical bootstrap (`hierboot`), power theory
  (`power`), and the binormal two-level data generator (`dgp`).
- `tests/` — regression tests against the published case-study numbers plus
  synthetic sanity checks. Two tests are slow (a bit-exact replay of the
  published bootstrap stream and a coverage study, ~15–20 min together);
  the rest finish in seconds.
- `experiments/` — the simulation studies and the case study of the article.
- `results/` — the exact JSON outputs used in the article.
- `data/` — per-item OOD scores of the four case-study training runs and the
  published summary table the regression tests check against.

## Reproducing the article

| Article object | Command | Time |
|---|---|---|
| Case study (Tables 1–2, retrospective power, n* = 12) | `python experiments/case_study.py` | ~5 min (`--fast` skips the simulations) |
| Power validation grid (270 cells, both boundaries of the null) | `python experiments/sim_power.py` | ~55 min on 12 cores |
| Non-Gaussian robustness grid (96 cells) | `python experiments/sim_power_robust.py` | ~13 min |
| Coverage map (31 cells) | `python experiments/sim_coverage_ext.py` | ~3.5 h |
| Six-cell coverage probe (superseded by the map) | `python experiments/sim_coverage.py` | ~40 min |
| Figures 1–3 | `python experiments/build_figures.py` | seconds |

All root seeds are fixed inside the scripts; re-running a study reproduces
the shipped JSON in `results/` up to floating-point identity.

## Data

`data/ood_scores/` holds the raw per-item scores of the four training runs
of the case study — a joint energy-based model on CIFAR-10, trained with the
SGLD and predictor–corrector samplers, two seeds each — on the CIFAR-10 test
set (in-distribution) and five OOD sets (SVHN, CIFAR-100, DTD, LSUN-R, iSUN).
`data/hier_bootstrap_tost.json` is the summary table published with the case
study; `tests/` reproduce its numbers bit-exactly from the raw scores.

## Licence

MIT.
