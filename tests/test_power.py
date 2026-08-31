"""Tests for equivtest.power: analytic TOST power and the required-n formula.

Contents:
  (a) monotonicity: power grows with n and with the margin, falls with |Delta|;
  (b) agreement with empirical power from full two-level simulation
      (rtol 0.05 at n >= 5);
  (c) seeds_needed is the inverse of tost_power (round-trip);
  plus the boundary property power(Delta -> margin) <= alpha.
"""
import numpy as np
import pytest

from equivtest.auroc import auroc, delong_variance
from equivtest.dgp import simulate_runs
from equivtest.power import seeds_needed, tost_power
from equivtest.tost import welch_tost

MARGIN = 0.01
ALPHA = 0.05


class TestMonotonicity:
    def test_power_increases_with_n(self):
        powers = [
            tost_power(
                delta_true=0.0, margin=MARGIN, n=n, sigma_seed=0.005, v_bar=2e-5,
                alpha=ALPHA,
            )
            for n in (3, 5, 8, 12, 20)
        ]
        assert all(b > a for a, b in zip(powers, powers[1:]))

    def test_power_increases_with_margin(self):
        powers = [
            tost_power(
                delta_true=0.0, margin=margin, n=6, sigma_seed=0.005, v_bar=2e-5,
                alpha=ALPHA,
            )
            for margin in (0.005, 0.01, 0.02, 0.04)
        ]
        assert all(b > a for a, b in zip(powers, powers[1:]))

    def test_power_falls_with_true_shift(self):
        powers = [
            tost_power(
                delta_true=d, margin=MARGIN, n=8, sigma_seed=0.005, v_bar=2e-5,
                alpha=ALPHA,
            )
            for d in (0.0, 0.003, 0.006, 0.009)
        ]
        assert all(b < a for a, b in zip(powers, powers[1:]))

    def test_power_at_the_margin_collapses_to_alpha(self):
        power = tost_power(
            delta_true=MARGIN, margin=MARGIN, n=50, sigma_seed=0.002, v_bar=1e-6,
            alpha=ALPHA,
        )
        assert power <= ALPHA + 0.005


class TestSeedsNeeded:
    @pytest.mark.parametrize("target", [0.5, 0.8, 0.9])
    @pytest.mark.parametrize("sigma_seed,v_bar", [(0.003, 1e-5), (0.006, 3e-5)])
    def test_round_trip_with_tost_power(self, target, sigma_seed, v_bar):
        n_star = seeds_needed(
            margin=MARGIN, power=target, sigma_seed=sigma_seed, v_bar=v_bar,
            alpha=ALPHA,
        )
        achieved = tost_power(
            delta_true=0.0, margin=MARGIN, n=n_star, sigma_seed=sigma_seed,
            v_bar=v_bar, alpha=ALPHA,
        )
        assert achieved >= target
        if n_star > 2:
            below = tost_power(
                delta_true=0.0, margin=MARGIN, n=n_star - 1, sigma_seed=sigma_seed,
                v_bar=v_bar, alpha=ALPHA,
            )
            assert below < target

    def test_returns_at_least_two(self):
        # One run per method cannot feed a Welch test.
        assert seeds_needed(margin=0.5, power=0.5, sigma_seed=1e-4, v_bar=1e-8) >= 2


class TestAgainstEmpiricalPower:
    def test_matches_full_two_level_simulation(self):
        # Design chosen so that the analytic power is ~0.7-0.8 at n=6:
        # both methods equal (Delta = 0), moderate seed and image noise.
        rng = np.random.default_rng(0)
        mu, sigma_seed, n, m = 0.6, 0.004, 6, 8300
        n_rep = 2000

        rejections = 0
        v_estimates = []
        for _ in range(n_rep):
            runs_a = simulate_runs(mu, sigma_seed, n, m, m, rng)
            runs_b = simulate_runs(mu, sigma_seed, n, m, m, rng)
            theta_a = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_a])
            theta_b = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_b])
            res = welch_tost(theta_a, theta_b, margin=MARGIN, alpha=ALPHA)
            rejections += int(res.equivalent)
            v_estimates.append(
                np.mean(
                    [delong_variance(r.id_scores, r.ood_scores)[0]
                     for r in runs_a + runs_b]
                )
            )
        empirical_power = rejections / n_rep
        analytic_power = tost_power(
            delta_true=0.0, margin=MARGIN, n=n, sigma_seed=sigma_seed,
            v_bar=float(np.mean(v_estimates)), alpha=ALPHA,
        )
        assert analytic_power == pytest.approx(empirical_power, rel=0.05)
