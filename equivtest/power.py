"""Analytic power of the seed-level Welch-TOST and the required run count.

Design model: each method is trained n times; the per-run AUROC
estimate has variance sigma_seed^2 + v_bar (between-seed dispersion plus the
mean DeLong image-level variance), so the standard error of the method
difference is SE = sqrt(2 (sigma_seed^2 + v_bar) / n) for equal group sizes,
with Welch-Satterthwaite df = 2 (n - 1).

Normal approximation of TOST power at true difference Delta:
  pi(Delta) = Phi((margin - Delta)/SE - z_{1-alpha})
            + Phi((margin + Delta)/SE - z_{1-alpha}) - 1,
truncated at zero. For the small n typical of ML experiments the z-quantile
and Phi are replaced by their central-t counterparts at df = 2(n - 1)
(shifted-t approximation); `seeds_needed` inverts the formula by iterating
over n, which also iterates the df.
"""
from __future__ import annotations

from scipy.stats import norm
from scipy.stats import t as student_t


def _se_delta(n: int, sigma_seed: float, v_bar: float) -> float:
    return (2.0 * (sigma_seed**2 + v_bar) / n) ** 0.5


def tost_power(
    delta_true: float,
    margin: float,
    n: int,
    sigma_seed: float,
    v_bar: float,
    alpha: float = 0.05,
    use_t: bool = True,
) -> float:
    """Approximate probability that TOST declares equivalence.

    n is the number of runs PER METHOD; sigma_seed is a standard deviation,
    v_bar a variance. With use_t=False the pure normal approximation is
    returned (the closed form quoted in the manuscript).
    """
    if n < 2:
        return 0.0
    se = _se_delta(n, sigma_seed, v_bar)
    if use_t:
        df = 2 * (n - 1)
        crit = student_t.ppf(1.0 - alpha, df=df)
        cdf = lambda x: student_t.cdf(x, df=df)
    else:
        crit = norm.ppf(1.0 - alpha)
        cdf = norm.cdf
    power = (
        cdf((margin - delta_true) / se - crit)
        + cdf((margin + delta_true) / se - crit)
        - 1.0
    )
    return float(max(0.0, min(1.0, power)))


def seeds_needed(
    margin: float,
    power: float,
    sigma_seed: float,
    v_bar: float,
    alpha: float = 0.05,
    delta_true: float = 0.0,
    n_max: int = 100_000,
) -> int:
    """Smallest number of runs per method giving at least the target power.

    Iterates n upwards (the df iterates with it). Raises ValueError when the
    target is unreachable, e.g. |delta_true| >= margin.
    """
    if abs(delta_true) >= margin:
        raise ValueError("power target unreachable: |delta_true| >= margin")
    for n in range(2, n_max + 1):
        if tost_power(delta_true, margin, n, sigma_seed, v_bar, alpha) >= power:
            return n
    raise ValueError(f"power target not reached by n_max={n_max}")
