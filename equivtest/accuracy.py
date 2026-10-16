"""Per-run accuracy on a shared test set and its two-level summaries.

Accuracy takes the place of AUROC in the second case study of the
accompanying article. A run's accuracy is the mean of its 0/1 correctness
vector over the m test items; conditional on the run the item-level variance
of that mean is binomial, p(1 - p)/m; the dependence between runs induced by
the shared test set is summarised by the mean pairwise Pearson correlation of
the correctness vectors and by the shares of items on which all runs agree.
"""
from __future__ import annotations

import numpy as np


def accuracy(correct: np.ndarray) -> float:
    """Mean of a 0/1 correctness vector."""
    c = np.asarray(correct, dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("correct must be a non-empty 1-d vector")
    return float(c.mean())


def binomial_variance(p: float, m: int) -> float:
    """p(1 - p)/m: item-level variance of a run's accuracy conditional on the run."""
    if m < 1:
        raise ValueError("m must be a positive number of items")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return float(p * (1.0 - p) / m)


def between_run_correlation(correct: np.ndarray) -> float:
    """Mean pairwise Pearson correlation between the rows of an R x m 0/1 matrix.

    Uses the identity sum_{i<j} r_ij = (||sum_i z_i||^2 - R) / 2 for the
    standardised rows z_i, so no R x R matrix is formed. Rows with zero
    variance (a run correct on every item or on none) have no defined
    correlation and are dropped; nan is returned when fewer than two rows
    remain.
    """
    x = np.asarray(correct, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("correct must be an R x m matrix")
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.sqrt((x**2).sum(axis=1))
    keep = norms > 0.0
    r = int(keep.sum())
    if r < 2:
        return float("nan")
    z = x[keep] / norms[keep, None]
    total = float((z.sum(axis=0) ** 2).sum())
    return (total - r) / (r * (r - 1))


def item_agreement(correct: np.ndarray) -> dict:
    """Shares of items always correct, never correct and mixed across the R rows."""
    x = np.asarray(correct)
    if x.ndim != 2:
        raise ValueError("correct must be an R x m matrix")
    col = x.astype(np.float64).mean(axis=0)
    always = float(np.mean(col == 1.0))
    never = float(np.mean(col == 0.0))
    return {
        "always_correct": always,
        "never_correct": never,
        "mixed": float(1.0 - always - never),
    }
