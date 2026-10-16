"""case2_common.py -- shared routines of the second case study.

Used by case2_modelzoo.py and case2_multiberts.py: moment summaries of
per-run accuracies, the design table n*(margin, power), and the three
resampling experiments (null-true split of one configuration with shift
alternatives, two-arm subsampling of a genuine comparison, and
pilot-then-design). All resampling is without replacement and disjoint
within a replication; every rate is returned with its Monte-Carlo standard
error sqrt(r(1 - r)/R).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, shapiro, skew
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equivtest.accuracy import binomial_variance  # noqa: E402
from equivtest.power import seeds_needed, tost_power  # noqa: E402
from equivtest.tost import welch_tost  # noqa: E402

ROOT_SEED = 20260921
ALPHA = 0.05
POWER_TARGETS = (0.80, 0.90)


def mc_se(rate: float, reps: int) -> float:
    return float(np.sqrt(rate * (1.0 - rate) / reps))


def tost_decision(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = ALPHA) -> tuple[bool, bool]:
    """(equivalent, degenerate): a draw with zero variance in both arms cannot be
    tested (accuracies are discrete, k/m) and is counted as not certified."""
    try:
        return welch_tost(a, b, margin, alpha).equivalent, False
    except ValueError:
        return False, True


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def software_versions() -> dict:
    import scipy

    out = {"python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__}
    try:
        import torch
        import torchvision

        out["torch"] = torch.__version__
        out["torchvision"] = torchvision.__version__
        out["cuda_device"] = (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return out


def today() -> str:
    return _dt.date.today().isoformat()


def moment_summary(acc: np.ndarray, m: int, rng: np.random.Generator | None = None,
                   shapiro_n: int = 500) -> dict:
    """Seed-level moments of per-run accuracies plus the item-level variances.

    v_r = p_r(1 - p_r)/m per run, v_bar their mean, and the ratio v_bar/s^2.
    Shapiro-Wilk is computed on a random subsample of at most shapiro_n runs.
    """
    acc = np.asarray(acc, dtype=np.float64)
    r = len(acc)
    s2 = float(acc.var(ddof=1))
    v_r = np.array([binomial_variance(float(p), m) for p in acc])
    v_bar = float(v_r.mean())
    out = {
        "n_runs": int(r), "m_items": int(m),
        "mean": float(acc.mean()), "sd": float(np.sqrt(s2)), "s2": s2,
        "skewness": float(skew(acc, bias=False)) if r > 2 else None,
        "excess_kurtosis": float(kurtosis(acc, fisher=True, bias=False)) if r > 3 else None,
        "min": float(acc.min()), "max": float(acc.max()),
        "deciles": {str(q): float(np.percentile(acc, q)) for q in range(10, 100, 10)},
        "v_bar": v_bar, "v_bar_over_s2": v_bar / s2 if s2 > 0 else None,
        "v_r_min": float(v_r.min()), "v_r_max": float(v_r.max()),
    }
    if r >= 3:
        if rng is None:
            rng = np.random.default_rng(ROOT_SEED)
        sub = acc if r <= shapiro_n else rng.choice(acc, size=shapiro_n, replace=False)
        w, p = shapiro(sub)
        out["shapiro"] = {"n": int(len(sub)), "W": float(w), "p": float(p)}
    return out


def design_table(s: float, margins, powers=POWER_TARGETS, alpha: float = ALPHA) -> dict:
    """n*(margin, power) = seeds_needed(margin, power, s, 0): shared test set, v_bar = 0."""
    return {str(d): {str(pw): int(seeds_needed(d, pw, s, 0.0, alpha)) for pw in powers}
            for d in margins}


def formula_power(s: float, margin: float, n: int, delta_true: float = 0.0,
                  alpha: float = ALPHA) -> dict:
    return {"t": tost_power(delta_true, margin, n, s, 0.0, alpha, use_t=True),
            "normal": tost_power(delta_true, margin, n, s, 0.0, alpha, use_t=False)}


def null_true_split(acc: np.ndarray, n_grid, margins, reps: int,
                    rng: np.random.Generator, alpha: float = ALPHA) -> dict:
    """Disjoint arms of size n from one configuration: Delta = 0 by construction.

    For every replication 2n distinct runs are drawn and split into arms A and
    B. Arm A is then shifted by 0, +margin and -margin; the TOST decision at
    each margin is recorded. Returns realised rates (certification at Delta = 0,
    size at the two boundaries), MC standard errors and the formula values.
    """
    acc = np.asarray(acc, dtype=np.float64)
    r = len(acc)
    s = float(acc.std(ddof=1))
    out = {"s": s, "reps": int(reps), "n_runs_available": int(r), "by_n": {}}
    for n in n_grid:
        if 2 * n > r:
            out["by_n"][str(n)] = {"skipped": f"2n = {2 * n} > R = {r}"}
            continue
        hits = {str(d): {"0": 0, "+": 0, "-": 0} for d in margins}
        degenerate = 0
        for _ in range(reps):
            idx = rng.choice(r, size=2 * n, replace=False)
            a, b = acc[idx[:n]], acc[idx[n:]]
            for d in margins:
                key = str(d)
                eq0, deg = tost_decision(a, b, d, alpha)
                hits[key]["0"] += int(eq0)
                hits[key]["+"] += int(tost_decision(a + d, b, d, alpha)[0])
                hits[key]["-"] += int(tost_decision(a - d, b, d, alpha)[0])
            degenerate += int(deg)
        row = {"degenerate_draws": degenerate}
        for d in margins:
            key = str(d)
            rates = {k: v / reps for k, v in hits[key].items()}
            row[key] = {
                "certification_rate_delta0": rates["0"],
                "mc_se_delta0": mc_se(rates["0"], reps),
                "size_at_plus_margin": rates["+"],
                "mc_se_plus": mc_se(rates["+"], reps),
                "size_at_minus_margin": rates["-"],
                "mc_se_minus": mc_se(rates["-"], reps),
                "formula": formula_power(s, d, n, 0.0, alpha),
            }
        out["by_n"][str(n)] = row
    return out


def first_n_reaching(by_n: dict, margin: str, target: float = 0.8):
    """Smallest grid n whose realised certification rate at Delta = 0 reaches target."""
    for n_key in sorted(by_n, key=int):
        row = by_n[n_key]
        if "skipped" in row:
            continue
        if row[margin]["certification_rate_delta0"] >= target:
            return int(n_key)
    return None


def two_arm_subsampling(acc_a: np.ndarray, acc_b: np.ndarray, n_grid, margins,
                        reps: int, rng: np.random.Generator, alpha: float = ALPHA) -> dict:
    """Certification rate of the TOST at n runs per method drawn without replacement.

    Draws are disjoint within each arm; arms are distinct configurations.
    Also returns the mean of the realised Delta-hat and the formula power at
    the full-data Delta-hat and at 0 with the pooled full-data s.
    """
    a_all = np.asarray(acc_a, dtype=np.float64)
    b_all = np.asarray(acc_b, dtype=np.float64)
    s_pooled = float(np.sqrt((a_all.var(ddof=1) + b_all.var(ddof=1)) / 2.0))
    delta_full = float(a_all.mean() - b_all.mean())
    out = {"s_pooled": s_pooled, "delta_full": delta_full, "reps": int(reps),
           "k_a": int(len(a_all)), "k_b": int(len(b_all)), "by_n": {}}
    for n in n_grid:
        if n > len(a_all) or n > len(b_all):
            out["by_n"][str(n)] = {"skipped": f"n = {n} exceeds the runs available"}
            continue
        hits = {str(d): 0 for d in margins}
        deltas = np.empty(reps)
        degenerate = 0
        for i in range(reps):
            a = a_all[rng.choice(len(a_all), size=n, replace=False)]
            b = b_all[rng.choice(len(b_all), size=n, replace=False)]
            deltas[i] = a.mean() - b.mean()
            for d in margins:
                eq, deg = tost_decision(a, b, d, alpha)
                hits[str(d)] += int(eq)
            degenerate += int(deg)
        row = {"mean_delta_hat": float(deltas.mean()), "sd_delta_hat": float(deltas.std(ddof=1)),
               "degenerate_draws": degenerate}
        for d in margins:
            rate = hits[str(d)] / reps
            row[str(d)] = {
                "certification_rate": rate, "mc_se": mc_se(rate, reps),
                "formula_at_delta_full": formula_power(s_pooled, d, n, delta_full, alpha),
                "formula_at_zero": formula_power(s_pooled, d, n, 0.0, alpha),
            }
        out["by_n"][str(n)] = row
    return out


def pilot_then_design(acc_a: np.ndarray, acc_b: np.ndarray, n0: int, margins,
                      reps: int, rng: np.random.Generator, power: float = 0.8,
                      alpha: float = ALPHA) -> dict:
    """n0 pilot runs per method -> pooled s -> n* = seeds_needed(margin, power, s, 0).

    Returns the distribution of n* over pilots (median, 2.5 and 97.5
    percentiles, min, max) per margin, plus the full-data pooled s and its n*.
    """
    a_all = np.asarray(acc_a, dtype=np.float64)
    b_all = np.asarray(acc_b, dtype=np.float64)
    s_full = float(np.sqrt((a_all.var(ddof=1) + b_all.var(ddof=1)) / 2.0))
    out = {"n0": int(n0), "reps": int(reps), "power": power, "s_full": s_full,
           "n_star_full": {str(d): int(seeds_needed(d, power, s_full, 0.0, alpha))
                           for d in margins},
           "by_margin": {}}
    if n0 > len(a_all) or n0 > len(b_all):
        out["skipped"] = f"n0 = {n0} exceeds the runs available"
        return out
    s_pilot = np.empty(reps)
    for i in range(reps):
        a = a_all[rng.choice(len(a_all), size=n0, replace=False)]
        b = b_all[rng.choice(len(b_all), size=n0, replace=False)]
        s_pilot[i] = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    out["s_pilot"] = {"median": float(np.median(s_pilot)),
                      "p2.5": float(np.percentile(s_pilot, 2.5)),
                      "p97.5": float(np.percentile(s_pilot, 97.5))}
    for d in margins:
        n_star = np.array([seeds_needed(d, power, float(s), 0.0, alpha) if s > 0 else 2
                           for s in s_pilot], dtype=np.int64)
        out["by_margin"][str(d)] = {
            "median": float(np.median(n_star)),
            "p2.5": float(np.percentile(n_star, 2.5)),
            "p97.5": float(np.percentile(n_star, 97.5)),
            "min": int(n_star.min()), "max": int(n_star.max()),
            "share_at_least_full_n_star": float(np.mean(n_star >= out["n_star_full"][str(d)])),
        }
    return out


def paired_tost(diff: np.ndarray, margin: float, alpha: float = ALPHA) -> dict:
    """One-sample t-based TOST on paired differences (sensitivity check only)."""
    d = np.asarray(diff, dtype=np.float64)
    k = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(k))
    df = k - 1
    if se == 0.0:
        return {"delta": mean, "se": 0.0, "df": df, "p_tost": None, "ci90": [mean, mean],
                "equivalent": False, "k": k, "degenerate": True}
    p_lower = float(1.0 - student_t.cdf((mean + margin) / se, df=df))
    p_upper = float(student_t.cdf((mean - margin) / se, df=df))
    t_crit = float(student_t.ppf(1.0 - alpha, df=df))
    return {"delta": mean, "se": se, "df": df, "p_tost": max(p_lower, p_upper),
            "ci90": [mean - t_crit * se, mean + t_crit * se],
            "equivalent": bool(p_lower < alpha and p_upper < alpha), "k": k}


def tost_dict(theta_a: np.ndarray, theta_b: np.ndarray, margin: float,
              alpha: float = ALPHA) -> dict:
    res = welch_tost(theta_a, theta_b, margin, alpha)
    return {"delta": res.delta, "se": res.se, "df": res.df, "p_tost": res.p_tost,
            "ci90": [res.ci90[0], res.ci90[1]], "equivalent": res.equivalent,
            "n_a": res.n_a, "n_b": res.n_b, "margin": margin}


def fmt_rate(rate: float, se: float) -> str:
    return f"{rate:.3f} ({se:.3f})"
