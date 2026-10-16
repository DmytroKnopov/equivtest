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

`tost_power_exact` evaluates the exact power of the pooled TOST under the
same normal working model (Shieh 2016, one-dimensional integral over the
pooled variance); it is the benchmark against which the heuristic is
compared in the manuscript, and `seeds_needed(..., exact=True)` inverts it.
"""
from __future__ import annotations

from scipy.integrate import quad
from scipy.stats import chi2, norm
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


def tost_power_exact(
    delta_true: float,
    margin: float,
    n: int,
    sigma_seed: float,
    v_bar: float,
    alpha: float = 0.05,
) -> float:
    """Exact TOST power under the pooled, equal-variance normal working model.

    The per-run values of the two methods are N(mu_A, tau^2) and
    N(mu_B, tau^2) with tau^2 = sigma_seed^2 + v_bar, n runs each. The pooled
    variance s^2 has nu = 2(n - 1) degrees of freedom and is independent of
    the mean difference D ~ N(Delta, 2 tau^2 / n). The pooled TOST declares
    equivalence iff -margin + t s c < D < margin - t s c, with c = sqrt(2/n)
    and t = t_{1-alpha, nu}; the event is empty once s >= margin / (t c).
    Conditioning on W = nu s^2 / tau^2 ~ chi^2_nu and integrating over W
    gives the one-dimensional integral of Shieh (2016), evaluated here by
    adaptive quadrature. This is the quantity that the shifted-t heuristic
    `tost_power` approximates: the heuristic replaces the noncentral-t laws
    with a shared random denominator by a shifted central t.
    """
    if n < 2:
        return 0.0
    nu = 2 * (n - 1)
    tau = (sigma_seed**2 + v_bar) ** 0.5
    c = (2.0 / n) ** 0.5
    t_crit = student_t.ppf(1.0 - alpha, df=nu)
    sd_diff = tau * c
    w_max = nu * (margin / (t_crit * c * tau)) ** 2

    def integrand(w: float) -> float:
        half_width = margin - t_crit * c * tau * (w / nu) ** 0.5
        prob = norm.cdf((half_width - delta_true) / sd_diff) - norm.cdf(
            (-half_width - delta_true) / sd_diff
        )
        return prob * chi2.pdf(w, df=nu)

    lo = chi2.ppf(1e-12, df=nu)
    hi = min(w_max, chi2.ppf(1.0 - 1e-12, df=nu))
    if hi <= lo:
        return 0.0
    breakpoints = [nu] if lo < nu < hi else None
    value, _ = quad(integrand, lo, hi, limit=200, points=breakpoints)
    return float(max(0.0, min(1.0, value)))


def seeds_needed(
    margin: float,
    power: float,
    sigma_seed: float,
    v_bar: float,
    alpha: float = 0.05,
    delta_true: float = 0.0,
    n_max: int = 100_000,
    exact: bool = False,
) -> int:
    """Smallest number of runs per method giving at least the target power.

    Iterates n upwards (the df iterates with it). With exact=True the exact
    pooled-model power `tost_power_exact` is inverted instead of the
    shifted-t heuristic. Raises ValueError when the target is unreachable,
    e.g. |delta_true| >= margin.
    """
    if abs(delta_true) >= margin:
        raise ValueError("power target unreachable: |delta_true| >= margin")
    power_fn = tost_power_exact if exact else tost_power
    for n in range(2, n_max + 1):
        if power_fn(delta_true, margin, n, sigma_seed, v_bar, alpha) >= power:
            return n
    raise ValueError(f"power target not reached by n_max={n_max}")
