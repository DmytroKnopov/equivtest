# Paired analysis of the eight seed-matched comparisons

| comparison | pairs | delta-hat (exact, 4 dp) | paired 90 % CI | paired p 0.01 | paired p 0.02 | verdict 0.01 / 0.02 | corr | unpaired p 0.01 / 0.02 | unpaired 90 % CI | n* (indep. arms, 0.01) | paired rate n=2 (degenerate) | n=3 (degenerate) | n=5 (degenerate) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zoo Adam vs SGD | 10 | -0.0002 | [-0.0080, +0.0077] | 0.024 | 0.001 | yes / yes | -0.229 | 0.010 / 0.000 | [-0.0069, +0.0066] | 14 | 0.0444 (exact, 45 subsets, 0 degenerate) | 0.0333 (exact, 120 subsets, 0 degenerate) | 0.0714 (exact, 252 subsets, 0 degenerate) |
| zoo GELU vs tanh | 10 | +0.0012 | [-0.0056, +0.0081] | 0.021 | 0.000 | yes / yes | -0.439 | 0.008 / 0.000 | [-0.0044, +0.0069] | 10 | 0.0222 (exact, 45 subsets, 0 degenerate) | 0.0667 (exact, 120 subsets, 0 degenerate) | 0.0913 (exact, 252 subsets, 0 degenerate) |
| RTE lr | 25 | +0.0983 | [+0.0737, +0.1229] | 1.000 | 1.000 | no / no | -0.151 | 1.000 / 1.000 | [+0.0754, +0.1213] | 397 | 0.0000 (exact, 300 subsets, 3 degenerate) | 0.0000 (exact, 2300 subsets, 0 degenerate) | 0.0000 (exact, 53130 subsets, 0 degenerate) |
| SST-2 lr | 25 | +0.0084 | [+0.0067, +0.0101] | 0.059 | 0.000 | no / yes | +0.483 | 0.125 / 0.000 | [+0.0061, +0.0107] | 6 | 0.0433 (exact, 300 subsets, 18 degenerate) | 0.0365 (exact, 2300 subsets, 11 degenerate) | 0.0759 (exact, 53130 subsets, 1 degenerate) |
| MNLI lr | 25 | +0.0040 | [+0.0028, +0.0052] | 0.000 | 0.000 | yes / yes | -0.057 | 0.000 / 0.000 | [+0.0028, +0.0051] | 3 | 0.2767 (exact, 300 subsets, 1 degenerate) | 0.6700 (exact, 2300 subsets, 0 degenerate) | 0.9876 (exact, 53130 subsets, 0 degenerate) |
| RTE steps | 5 | +0.0404 | [-0.0134, +0.0943] | 0.853 | 0.768 | no / no | +0.804 | 0.758 / 0.683 | [-0.0393, +0.1201] | 705 | - | - | - |
| SST-2 steps | 5 | -0.0101 | [-0.0137, -0.0065] | 0.520 | 0.002 | no / yes | -0.430 | 0.525 / 0.000 | [-0.0127, -0.0075] | 3 | - | - | - |
| MNLI steps | 5 | -0.0094 | [-0.0114, -0.0073] | 0.281 | 0.000 | no / yes | -0.508 | 0.233 / 0.000 | [-0.0109, -0.0079] | 2 | - | - | - |
