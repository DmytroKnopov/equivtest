# Case study 2: CIFAR-10 Model Zoos (Schuerholt et al., 2022)

Date 2026-09-21, root seed 20260921, 5000 replications, device cuda, runtime 1.8 min. Sign convention: delta = mean(A) - mean(B). Test set m = 10000.

## Data verification

| Zoo | Runs | max abs diff | beyond 1e-3 | beyond 1e-4 | exact |
|---|---:|---:|---:|---:|---:|
| seed zoo | 1000 | 0.00010 | 0 | 0 | 999 |
| hyperparameter zoo (selected nodes) | 30 | 0.00010 | 0 | 0 | 29 |

## Analysis A: seed zoo, one configuration

| R | mean | s | skewness | excess kurtosis | min | max | Shapiro p (n=500) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.4872 | 0.01358 | -1.800 | +5.397 | 0.4116 | 0.5133 | 1.96e-20 |

| decile | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 0.4704 | 0.4794 | 0.4842 | 0.4874 | 0.4897 | 0.4920 | 0.4945 | 0.4971 | 0.5008 |

- v_bar = 2.497e-05 (range of v_r [2.422e-05, 2.500e-05]), s^2 = 1.844e-04, v_bar / s^2 = 0.135
- between-run correlation of correctness: 0.5737 (all runs; 200-run subsample 0.5748)
- items always correct 0.0384, never correct 0.0529, mixed 0.9087

| margin | n* (80%) | n* (90%) |
|---:|---:|---:|
| 0.005 | 128 | 161 |
| 0.01 | 33 | 41 |
| 0.015 | 15 | 19 |
| 0.02 | 9 | 11 |

## Analysis B: real seed law (disjoint arms of n runs, Delta = 0 by construction)

Margins: 0.01 and 0.0173 (calibrated so that n*(80%) = 12; interval [0.016845, 0.017660]). Rates with MC standard errors.

### margin 0.01: n*(80%) = 33, n*(90%) = 41, first grid n with realised rate >= 0.8: 50

| n | certification rate (Delta=0) | formula (t) | formula (normal) | size at +margin | size at -margin |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.014 (0.002) | 0.000 | 0.000 | 0.008 (0.001) | 0.011 (0.001) |
| 3 | 0.029 (0.002) | 0.000 | 0.000 | 0.010 (0.001) | 0.008 (0.001) |
| 5 | 0.052 (0.003) | 0.000 | 0.000 | 0.012 (0.002) | 0.012 (0.002) |
| 8 | 0.115 (0.005) | 0.000 | 0.000 | 0.024 (0.002) | 0.024 (0.002) |
| 12 | 0.243 (0.006) | 0.068 | 0.126 | 0.030 (0.002) | 0.028 (0.002) |
| 20 | 0.502 (0.007) | 0.476 | 0.506 | 0.046 (0.003) | 0.047 (0.003) |
| 50 | 0.951 (0.003) | 0.954 | 0.958 | 0.041 (0.003) | 0.055 (0.003) |

### margin 0.0173: n*(80%) = 12, n*(90%) = 15, first grid n with realised rate >= 0.8: 20

| n | certification rate (Delta=0) | formula (t) | formula (normal) | size at +margin | size at -margin |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.067 (0.004) | 0.000 | 0.000 | 0.017 (0.002) | 0.018 (0.002) |
| 3 | 0.168 (0.005) | 0.000 | 0.000 | 0.026 (0.002) | 0.025 (0.002) |
| 5 | 0.380 (0.007) | 0.119 | 0.288 | 0.034 (0.003) | 0.035 (0.003) |
| 8 | 0.604 (0.007) | 0.555 | 0.634 | 0.043 (0.003) | 0.045 (0.003) |
| 12 | 0.794 (0.006) | 0.826 | 0.860 | 0.046 (0.003) | 0.045 (0.003) |
| 20 | 0.963 (0.003) | 0.976 | 0.983 | 0.049 (0.003) | 0.052 (0.003) |
| 50 | 1.000 (0.000) | 1.000 | 1.000 | 0.041 (0.003) | 0.055 (0.003) |

## Analysis C: two-method comparisons (hyperparameter zoo, 10 seeds per node)

### P1_adam_vs_sgd: init=kaiming_uniform,nlin=gelu,opt=adam,wd=0.01,dropout=0,lr=0.001 vs init=kaiming_uniform,nlin=gelu,opt=sgd,wd=0.01,dropout=0,lr=0.001

- mean A 0.5024 (sd 0.00905), mean B 0.5026 (sd 0.00826), pooled s 0.00867; correlation of correctness within A 0.580, within B 0.584, between nodes 0.579

| margin | delta | 90% CI | p | equivalent | n*(80%) full s | rate n=2 | rate n=3 | rate n=5 | pilot n* median [2.5%, 97.5%] |
|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| 0.01 | -0.0002 | [-0.0069, +0.0066] | 0.0103 | True | 14 | 0.026 (0.002) | 0.066 (0.004) | 0.112 (0.004) | 14 [5, 25] |
| 0.02 | -0.0002 | [-0.0069, +0.0066] | 0.0000 | True | 5 | 0.188 (0.006) | 0.440 (0.007) | 0.995 (0.001) | 5 [3, 7] |

### P2_gelu_vs_tanh: init=kaiming_uniform,nlin=gelu,opt=adam,wd=0.01,dropout=0,lr=0.001 vs init=kaiming_uniform,nlin=tanh,opt=adam,wd=0.01,dropout=0,lr=0.001

- mean A 0.5024 (sd 0.00905), mean B 0.5011 (sd 0.00450), pooled s 0.00715; correlation of correctness within A 0.580, within B 0.639, between nodes 0.533

| margin | delta | 90% CI | p | equivalent | n*(80%) full s | rate n=2 | rate n=3 | rate n=4 | rate n=5 | rate n=10 | pilot n* median [2.5%, 97.5%] |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.01 | +0.0012 | [-0.0044, +0.0069] | 0.0084 | True | 10 | 0.031 (0.002) | 0.084 (0.004) | 0.182 (0.005) | 0.300 (0.006) | 1.000 (0.000) | 10 [5, 16] |
| 0.02 | +0.0012 | [-0.0044, +0.0069] | 0.0000 | True | 4 | 0.233 (0.006) | 0.637 (0.007) | 0.970 (0.002) | 1.000 (0.000) | 1.000 (0.000) | 4 [3, 5] |

