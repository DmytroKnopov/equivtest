"""Two-level (seed x image) hierarchical bootstrap for method comparison.

Per bootstrap iteration the resampling and the RNG draw order are:
(1) seed indices for the first group, (2) seed indices for
the second group, (3) one shared (paired) resample of ID image indices,
(4) one shared resample of OOD image indices per dataset. Runs are objects
with `id_scores` (array) and `ood_scores` (array, or dict {dataset: array}
resampled in insertion order).

Sign convention: Delta = stat(runs_a) - stat(runs_b). Passing
(sgld_runs, pc_runs) replays the resampling stream of the published case
study; the resulting bootstrap distribution is the negation of the published
PC - SGLD one, which took Delta = second group - first group.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from equivtest.auroc import auroc


def _ood_dict(run) -> dict:
    ood = run.ood_scores
    return ood if isinstance(ood, dict) else {"ood": ood}


def _default_stat(aurocs: dict) -> float:
    """Macro-average AUROC over the OOD suite (macro-5 in the case study)."""
    return float(np.mean(list(aurocs.values())))


def hierarchical_ci(
    runs_a: list,
    runs_b: list,
    stat_fn: Callable[[dict], float] = _default_stat,
    B: int = 3000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray]:
    """Percentile CI for Delta = stat(runs_a) - stat(runs_b).

    Outer level: seeds are resampled with replacement within each method
    (n_g draws from n_g runs). Inner level: image indices are resampled once
    per iteration and shared across all selected runs of both methods
    (paired design: every run is scored on the same resampled test sets).
    Returns (ci_low, ci_high, bootstrap_distribution).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_a, n_b = len(runs_a), len(runs_b)
    datasets = list(_ood_dict(runs_a[0]).keys())
    m0 = len(runs_a[0].id_scores)
    m_out = {ds: len(_ood_dict(runs_a[0])[ds]) for ds in datasets}

    def run_stat(run, idx_in, idx_out) -> float:
        in_b = run.id_scores[idx_in]
        ood = _ood_dict(run)
        aurocs = {ds: auroc(in_b, ood[ds][idx_out[ds]]) for ds in datasets}
        return stat_fn(aurocs)

    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        idx_in = rng.integers(0, m0, size=m0)
        idx_out = {ds: rng.integers(0, m_out[ds], size=m_out[ds]) for ds in datasets}
        stats_a = [run_stat(runs_a[i], idx_in, idx_out) for i in idx_a]
        stats_b = [run_stat(runs_b[i], idx_in, idx_out) for i in idx_b]
        boot[b] = np.mean(stats_a) - np.mean(stats_b)

    ci_low = float(np.quantile(boot, alpha / 2))
    ci_high = float(np.quantile(boot, 1 - alpha / 2))
    return ci_low, ci_high, boot
