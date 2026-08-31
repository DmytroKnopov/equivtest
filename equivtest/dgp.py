"""Binormal two-level (seed x image) data-generating process.

Model: a training run s of a method with mean
performance mu has true AUROC theta_s = mu + a_s, a_s ~ N(0, sigma_seed^2),
redrawn if theta_s falls outside (0, 1) so the AUROC stays a probability;
conditional on the run, ID scores are N(0, 1) and OOD scores are
N(m_s, 1) with the binormal link theta = Phi(m / sqrt(2)), i.e.
m_s = sqrt(2) * Phi^{-1}(theta_s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Run:
    """One simulated training run: per-image scores and its true AUROC."""

    id_scores: np.ndarray
    ood_scores: np.ndarray
    theta_true: float


def theta_from_shift(shift: float) -> float:
    """Binormal link: theta = Phi(shift / sqrt(2))."""
    return float(norm.cdf(shift / np.sqrt(2.0)))


def shift_from_theta(theta: float) -> float:
    """Inverse binormal link: shift = sqrt(2) * Phi^{-1}(theta)."""
    return float(np.sqrt(2.0) * norm.ppf(theta))


_MAX_THETA_TRIES = 1000


def _check_params(mu: float, sigma_seed: float) -> None:
    """Validate generator parameters: mu a proper AUROC, sigma_seed a scale."""
    if not 0.0 < mu < 1.0:
        raise ValueError(f"mu must lie strictly inside (0, 1), got {mu}")
    if not (np.isfinite(sigma_seed) and sigma_seed >= 0.0):
        raise ValueError(
            f"sigma_seed must be finite and non-negative, got {sigma_seed}"
        )


def _draw_theta(mu: float, sigma_seed: float, rng: np.random.Generator) -> float:
    """One run-level AUROC draw, redrawn while outside (0, 1)."""
    for _ in range(_MAX_THETA_TRIES):
        theta_s = mu + sigma_seed * rng.standard_normal()
        if 0.0 < theta_s < 1.0:
            return float(theta_s)
    raise RuntimeError(
        f"no theta draw landed in (0, 1) after {_MAX_THETA_TRIES} tries "
        f"(mu={mu}, sigma_seed={sigma_seed})"
    )


def simulate_runs(
    mu: float,
    sigma_seed: float,
    n_seeds: int,
    m0: int,
    m1: int,
    rng: np.random.Generator,
) -> list[Run]:
    """Simulate n_seeds independent runs of one method.

    Each run draws theta_s ~ N(mu, sigma_seed^2), redrawn if outside (0, 1)
    to keep the binormal link defined, then m0 ID scores from N(0, 1) and
    m1 OOD scores from N(shift_from_theta(theta_s), 1).
    """
    _check_params(mu, sigma_seed)
    runs = []
    for _ in range(n_seeds):
        theta_s = _draw_theta(mu, sigma_seed, rng)
        shift = shift_from_theta(theta_s)
        id_scores = rng.standard_normal(m0)
        ood_scores = shift + rng.standard_normal(m1)
        runs.append(Run(id_scores, ood_scores, theta_s))
    return runs


def simulate_experiment(
    mu_a: float,
    mu_b: float,
    sigma_seed: float,
    n_seeds: int,
    m0: int,
    m1: int,
    rng: np.random.Generator,
    image_rho: float = 0.0,
) -> tuple[list[Run], list[Run]]:
    """Simulate a two-method experiment on a SHARED test set.

    Every run of both methods scores the same m0 + m1 test images; a score
    is sqrt(image_rho) * shared image effect + sqrt(1 - image_rho) * run
    noise, so image_rho is the cross-run correlation of same-image scores
    (~0.8-0.9 in the case-study data) and the marginal laws N(0, 1) /
    N(shift, 1) -- hence the binormal link -- are unchanged. image_rho=0
    recovers independent per-run scores as in simulate_runs.
    """
    _check_params(mu_a, sigma_seed)
    _check_params(mu_b, sigma_seed)
    w_shared = np.sqrt(image_rho)
    w_noise = np.sqrt(1.0 - image_rho)
    id_effects = rng.standard_normal(m0)
    ood_effects = rng.standard_normal(m1)

    def one_method(mu: float) -> list[Run]:
        runs = []
        for _ in range(n_seeds):
            theta_s = _draw_theta(mu, sigma_seed, rng)
            shift = shift_from_theta(theta_s)
            id_scores = w_shared * id_effects + w_noise * rng.standard_normal(m0)
            ood_scores = (
                shift + w_shared * ood_effects + w_noise * rng.standard_normal(m1)
            )
            runs.append(Run(id_scores, ood_scores, theta_s))
        return runs

    return one_method(mu_a), one_method(mu_b)
