"""AUROC as a two-sample Mann-Whitney U-statistic and its DeLong variance.

Conventions follow the OOD-detection setting of the companion case study:
`id_scores` are in-distribution scores, `ood_scores` are out-of-distribution
scores, higher score = more OOD, and AUROC = P(ood > id) with ties counted
as 1/2. Both the rank-based AUROC and the DeLong components use the
average-rank machinery of Sun & Xu (2014).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Mann-Whitney U-based AUROC: P(out > in), ties as 1/2."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    m0 = len(id_scores)
    m1 = len(ood_scores)
    combined = np.concatenate([id_scores, ood_scores])
    ranks = rankdata(combined, method="average")
    return float((ranks[m0:].sum() - m1 * (m1 + 1) / 2) / (m0 * m1))


def delong_variance(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> tuple[float, float, float]:
    """DeLong variance of the AUROC estimate for one run.

    Returns (v, s10, s01) where s10 is the sample variance of the placement
    values of OOD scores, s01 the same for ID scores, and
    v = s10/m1 + s01/m0 estimates Var(theta_hat) conditional on the run.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    m0 = len(id_scores)
    m1 = len(ood_scores)
    combined = np.concatenate([id_scores, ood_scores])
    ranks_all = rankdata(combined, method="average")
    ranks_id = rankdata(id_scores, method="average")
    ranks_ood = rankdata(ood_scores, method="average")
    # Placement values: fraction of the other sample below the given score
    # (ties as 1/2), recovered from the difference of combined and
    # within-sample average ranks.
    v01 = 1.0 - (ranks_all[:m0] - ranks_id) / m1  # per ID score: P(ood > id_i)
    v10 = (ranks_all[m0:] - ranks_ood) / m0       # per OOD score: P(ood_j > id)
    s10 = float(np.var(v10, ddof=1))
    s01 = float(np.var(v01, ddof=1))
    v = s10 / m1 + s01 / m0
    return v, s10, s01
