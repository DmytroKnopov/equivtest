"""Seed-level Welch-TOST for AUROC equivalence.

Two one-sided Welch t-tests on per-run (seed-level) AUROCs.
H0: |delta| >= margin; equivalence is declared only if both one-sided
tests reject at level alpha, which is dual to the (1 - 2 alpha) confidence
interval lying inside (-margin, +margin).

Sign convention: delta = mean(theta_a) - mean(theta_b).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass
class TostResult:
    delta: float
    se: float
    df: float
    margin: float
    alpha: float
    p_lower: float
    p_upper: float
    p_tost: float
    ci90: tuple[float, float]
    equivalent: bool
    n_a: int
    n_b: int


def welch_tost(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> TostResult:
    """Two one-sided tests (Schuirmann) with Welch-Satterthwaite df.

    theta_a, theta_b: per-run AUROC estimates of the two methods.
    Raises ValueError when the pooled standard error is zero (degenerate
    design with no between-seed variability).
    """
    a = np.asarray(theta_a, dtype=np.float64)
    b = np.asarray(theta_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    se = float(np.sqrt(var_a / n_a + var_b / n_b))
    if se == 0.0:
        raise ValueError("zero between-seed variance: TOST is undefined")
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a**2) / (n_a**2 * (n_a - 1)) + (var_b**2) / (n_b**2 * (n_b - 1))
    df = float(df_num / df_den)
    delta = float(a.mean() - b.mean())

    # H_lower: delta <= -margin, rejected for large (delta + margin) / se.
    t_lower = (delta + margin) / se
    p_lower = float(1.0 - student_t.cdf(t_lower, df=df))
    # H_upper: delta >= +margin, rejected for small (delta - margin) / se.
    t_upper = (delta - margin) / se
    p_upper = float(student_t.cdf(t_upper, df=df))
    p_tost = max(p_lower, p_upper)

    t_crit = float(student_t.ppf(1.0 - alpha, df=df))
    ci90 = (delta - t_crit * se, delta + t_crit * se)

    return TostResult(
        delta=delta,
        se=se,
        df=df,
        margin=margin,
        alpha=alpha,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        ci90=ci90,
        equivalent=bool(p_lower < alpha and p_upper < alpha),
        n_a=n_a,
        n_b=n_b,
    )
