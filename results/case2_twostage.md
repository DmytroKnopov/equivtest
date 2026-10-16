# Two-stage design carried through on the seed zoo

Seed zoo: 1000 runs, s = 0.01358; pilot n0 = 5 per method, target power 0.8, alpha 0.05, floor 5; 5000 replications (Gaussian reference 20000).

| margin | n* full | oracle rate at n* full | rule | achieved rate | n* median [2.5, 97.5] | share below n* full | mean total runs | Gaussian rate | Gaussian mean runs |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 33 | 0.799 ± 0.006 | plugin | 0.598 ± 0.007 | 24 [6, 116] | 0.662 | 77.1 | 0.692 ± 0.003 | 75.9 |
| 0.01 | 33 | 0.799 ± 0.006 | upper80 | 0.779 ± 0.006 | 39 [10, 189] |  | 121.4 | 0.884 ± 0.002 | 122.9 |
| 0.0173 | 12 | 0.794 ± 0.006 | plugin | 0.662 ± 0.007 | 9 [5, 38] | 0.660 | 34.0 | 0.718 ± 0.003 | 34.0 |
| 0.0173 | 12 | 0.794 ± 0.006 | upper80 | 0.802 ± 0.006 | 14 [5, 66] |  | 49.8 | 0.894 ± 0.002 | 49.4 |

Reference checks (20000 pilots): Gaussian point-estimate undersizing share 0.01: 0.562, 0.0173: 0.542; P(ten pilot runs miss the lowest 5 %) exact 0.597, empirical 0.599; upper 80 % bound (factor 1.742) covers the full-pool variance in 0.608 of pilots on the real seed law and 0.801 under a Gaussian law.
