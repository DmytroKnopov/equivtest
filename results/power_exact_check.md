# Shifted-t heuristic against the exact pooled-model TOST power, the Welch-TOST on normal data and the realised rates

Grid of sim_power.json, margin 0.01, alpha 0.05; 'inside' = cells with |Delta| < margin; 'remark range' = inside, n >= 5, heuristic >= 0.2; Welch-normal by Monte Carlo with 200000 replications per cell (root seed 20261007).

## Inside the margin, by n

| n | cells | max abs (heur - exact) | max (heur - exact) | max (exact - welch) | heur - welch min / max | emp - welch min / max | median abs (emp - welch) | max abs (emp - exact) | max abs z (emp vs welch) | cells abs z > 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 18 | 0.312 | -0.0209 | 0.205 | -0.183 / -0.004 | -0.014 / +0.012 | 0.0030 | 0.219 | 2.21 | 0 |
| 3 | 18 | 0.215 | -0.0186 | 0.087 | -0.169 / +0.052 | -0.011 / +0.011 | 0.0027 | 0.089 | 2.12 | 0 |
| 5 | 18 | 0.061 | -0.0057 | 0.027 | -0.054 / +0.012 | -0.014 / +0.009 | 0.0046 | 0.040 | 2.00 | 0 |
| 8 | 18 | 0.084 | -0.0002 | 0.007 | -0.077 / +0.002 | -0.013 / +0.010 | 0.0035 | 0.016 | 1.37 | 0 |
| 12 | 18 | 0.014 | -0.0000 | 0.004 | -0.012 / -0.000 | -0.015 / +0.020 | 0.0052 | 0.018 | 1.80 | 0 |
| 20 | 18 | 0.006 | -0.0000 | 0.003 | -0.006 / -0.000 | -0.015 / +0.010 | 0.0018 | 0.016 | 2.57 | 0 |

## Remark range (inside, n >= 5, heuristic >= 0.2), by n

| n | cells | max (exact - heur) | max (exact - welch) | heur - welch min / max | emp - welch min / max | max abs (emp - exact) |
|---|---|---|---|---|---|---|
| 5 | 12 | 0.045 | 0.027 | -0.028 / +0.012 | -0.014 / +0.008 | 0.040 |
| 8 | 12 | 0.018 | 0.007 | -0.012 / +0.002 | -0.013 / +0.007 | 0.016 |
| 12 | 18 | 0.014 | 0.004 | -0.012 / -0.000 | -0.015 / +0.020 | 0.018 |
| 20 | 18 | 0.006 | 0.003 | -0.006 / -0.000 | -0.015 / +0.010 | 0.016 |

Remark range overall: max (exact - heur) 0.0450; max (exact - welch) 0.0266; heur - welch from -0.0282 to +0.0123; emp - welch from -0.0155 to +0.0202 (max abs 0.0202, median abs 0.0045, max abs z 2.57, cells abs z > 3: 0); emp - exact from -0.0398 to +0.0174.
Inside, n >= 5 (all power levels): max |heur - exact| 0.0843, heur - welch from -0.0774 to +0.0123, max |emp - exact| 0.0398, max abs z (emp vs welch) 2.57, cells abs z > 3: 0.
Inside, n < 5: max |heur - exact| 0.3117, max |emp - exact| 0.2190, max (exact - welch) 0.2049, max abs z (emp vs welch) 2.21, cells abs z > 3: 0.
All 108 inside cells: max abs z (emp vs welch) 2.57, cells abs z > 3: 0.
Cells where the heuristic exceeds the exact power: 35 (Delta values [0.02]; max excess 0.00300; largest heuristic 0.00301, largest exact 0.000277; inside or at the boundary: 0).
Worst heuristic cell (n >= 5, inside): sigma_seed 0.01, n 8, theta 0.55, Delta 0.0: heur 0.072, exact 0.156, welch 0.149, emp 0.143
Worst empirical cell (n >= 5, inside): sigma_seed 0.005, n 5, theta 0.75, Delta 0.0: heur 0.528, exact 0.548, welch 0.523, emp 0.508

Boundary cells (|Delta| = margin): max exact / max empirical by n
- n = 2: exact 0.0500, empirical 0.0320
- n = 3: exact 0.0500, empirical 0.0410
- n = 5: exact 0.0500, empirical 0.0535
- n = 8: exact 0.0500, empirical 0.0565
- n = 12: exact 0.0500, empirical 0.0630
- n = 20: exact 0.0500, empirical 0.0560

## Table 2 (first case study, s = 0.0080, Delta = 0)

| margin | n* 80% heur | n* 80% exact | n* 90% heur | n* 90% exact | exact power at heur n (80%) |
|---|---|---|---|---|---|
| 0.005 | 45 | 45 | 56 | 56 | 0.810 |
| 0.0075 | 21 | 21 | 26 | 26 | 0.824 |
| 0.01 | 12 | 12 | 15 | 15 | 0.818 |
| 0.015 | 6 | 6 | 8 | 7 | 0.834 |
| 0.02 | 4 | 4 | 5 | 5 | 0.860 |

n = 2: heuristic 0.000, exact 0.0542 (at Delta-hat 0.0437), welch-normal 0.0303 +- 0.0002, empirical 0.026 +- 0.004
n = 12: heuristic 0.812, exact 0.818, welch-normal 0.8149 +- 0.0004, empirical 0.811 +- 0.009; at Delta-hat heuristic 0.352, exact 0.363
n* (80%) at Delta-hat: heuristic 43, exact 43

## Zoo two-stage margins (s_full = 0.0136)
- margin 0.01: published 33, heuristic 33, exact 33, exact power at published n 0.811, oracle realised 0.799
- margin 0.0173: published 12, heuristic 12, exact 12, exact power at published n 0.832, oracle realised 0.794

## Single-configuration design tables (Table 3 prints margin 0.01, power 0.8)
- CIFAR-10 zoo (s 0.0136): 0.005/0.8: 128 vs 128; 0.005/0.9: 161 vs 161; 0.01/0.8: 33 vs 33; 0.01/0.9: 41 vs 41; 0.015/0.8: 15 vs 15; 0.015/0.9: 19 vs 19; 0.02/0.8: 9 vs 9; 0.02/0.9: 11 vs 11
- MultiBERTs, RTE (s 0.0609): 0.01/0.8: 636 vs 635 (differs); 0.01/0.9: 803 vs 803; 0.02/0.8: 160 vs 160; 0.02/0.9: 202 vs 202
- MultiBERTs, SST-2 (s 0.0048): 0.01/0.8: 5 vs 5; 0.01/0.9: 7 vs 6 (differs); 0.02/0.8: 3 vs 3; 0.02/0.9: 3 vs 3
- MultiBERTs, MNLI (s 0.0021): 0.01/0.8: 3 vs 2 (differs); 0.01/0.9: 3 vs 3; 0.02/0.8: 2 vs 2; 0.02/0.9: 2 vs 2

## Table 4 planning column (independent arms, Delta = 0 unless stated)
- zoo Adam vs SGD: s 0.0087; 0.01: heur 14, exact 14, published 14; at_delta_hat_0.01: heur 14, exact 14
- zoo GELU vs tanh: s 0.0071; 0.01: heur 10, exact 10, published 10; at_delta_hat_0.01: heur 11, exact 11
- RTE lr: s 0.0480; 0.01: heur 397, exact 396, published 397
- SST-2 lr: s 0.0049; 0.01: heur 6, exact 5, published 6; at_delta_hat_0.01: heur 116, exact 116
- MNLI lr: s 0.0024; 0.01: heur 3, exact 3, published 3; at_delta_hat_0.01: heur 3, exact 3
- RTE steps: s 0.0641; 0.01: heur 705, exact 705, published 705
- SST-2 steps: s 0.0022; 0.01: heur 3, exact 3, published 3
- MNLI steps: s 0.0012; 0.01: heur 2, exact 2, published 2; at_delta_hat_0.01: heur 54, exact 54
