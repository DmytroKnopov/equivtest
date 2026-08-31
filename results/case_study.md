# Case study: the four training runs through the equivalence protocol

Margin 0.01, alpha 0.05. ID: CIFAR-10 test (m0=10000); OOD sizes: svhn 26032, cifar100 10000, dtd 1880, lsun 10000, isun 8925.

## Per-run AUROCs

| Run | SVHN | CIFAR100 | DTD | LSUN | ISUN | macro-5 |
|---|---:|---:|---:|---:|---:|---:|
| SGLD, seed 42 | 0.5676 | 0.6396 | 0.7250 | 0.7805 | 0.7863 | **0.6998** |
| SGLD, seed 123 | 0.6056 | 0.6244 | 0.7349 | 0.7688 | 0.7899 | **0.7047** |
| PC, seed 42 | 0.6045 | 0.6418 | 0.7374 | 0.7511 | 0.7668 | **0.7003** |
| PC, seed 123 | 0.6237 | 0.6391 | 0.7549 | 0.7737 | 0.7859 | **0.7155** |

SGLD mean 0.7022, PC mean 0.7079, delta (PC - SGLD) = +0.0056.

## Estimates

- s_SGLD = 0.0035, s_PC = 0.0107, pooled s = 0.0080 (s^2 = 6.343e-05)
- v_bar(macro-5) = 8.633e-06 = 13.6% of s^2
- between-run score correlation: cifar10 0.91, svhn 0.79, cifar100 0.86, dtd 0.92, lsun 0.91, isun 0.92

## TOST (reproduced)

- delta = +0.0056, SE = 0.0080, df = 1.21, p = 0.3342 -> no equivalence at margin 0.01
- Welch 90% CI [-0.0322, +0.0435]; published hierarchical 95% CI [-0.0046, +0.0161]

## Power

- retrospective (n=2, delta=0): t 0.0000, normal 0.0000
- empirical (calibrated proxy): n=2 -> 0.026 (MC SE 0.004); n=12 -> 0.811 (MC SE 0.009)
- n*(delta=0.01) = 12 (80%), 15 (90%); chi^2 95% interval on n*(80%): [4, 3542] (df = 1.21)

## Design table n*(margin, power)

| margin | 80% | 90% |
|---:|---:|---:|
| 0.005 | 45 | 56 |
| 0.0075 | 21 | 26 |
| 0.01 | 12 | 15 |
| 0.015 | 6 | 8 |
| 0.02 | 4 | 5 |
