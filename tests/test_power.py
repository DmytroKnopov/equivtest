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
from equivtest.power import seeds_needed, tost_power, tost_power_exact
from equivtest.tost import welch_tost

MARGIN = 0.01
ALPHA = 0.05


class TestExactPower:
    """tost_power_exact against a direct Monte-Carlo evaluation of the pooled
    normal model, and its relation to the shifted-t heuristic."""

    @pytest.mark.parametrize(
        "n,delta_true", [(2, 0.0), (3, 0.0), (5, 0.0), (5, 0.005), (12, 0.0), (12, 0.007)]
    )
    def test_matches_monte_carlo_of_the_normal_model(self, n, delta_true):
        sigma_seed, v_bar = 0.0075, 1e-5
        tau = (sigma_seed**2 + v_bar) ** 0.5
        nu = 2 * (n - 1)
        c = (2.0 / n) ** 0.5
        from scipy.stats import t as student_t

        t_crit = student_t.ppf(1 - ALPHA, df=nu)
        rng = np.random.default_rng(20261006)
        reps = 400_000
        diff = rng.normal(delta_true, tau * c, size=reps)
        s = tau * np.sqrt(rng.chisquare(nu, size=reps) / nu)
        declared = np.abs(diff) < MARGIN - t_crit * s * c
        mc = declared.mean()
        mc_se = (mc * (1 - mc) / reps) ** 0.5
        exact = tost_power_exact(delta_true, MARGIN, n, sigma_seed, v_bar, ALPHA)
        assert abs(exact - mc) < 4 * mc_se + 1e-4

    def test_positive_at_two_runs_where_the_heuristic_is_zero(self):
        # Case-study plug-in: s = 0.0080, margin 0.01, n = 2.
        heuristic = tost_power(0.0, MARGIN, 2, 0.0080, 0.0, ALPHA)
        exact = tost_power_exact(0.0, MARGIN, 2, 0.0080, 0.0, ALPHA)
        assert heuristic == 0.0
        assert 0.0 < exact < 0.1

    def test_converges_to_the_heuristic_for_large_n(self):
        # Margin scaled with n so that the power stays in the informative range.
        sigma_seed, v_bar = 0.0075, 1e-5
        tau = (sigma_seed**2 + v_bar) ** 0.5
        for n in (100, 300):
            margin = 2.5 * tau * (2.0 / n) ** 0.5
            exact = tost_power_exact(0.0, margin, n, sigma_seed, v_bar, ALPHA)
            heuristic = tost_power(0.0, margin, n, sigma_seed, v_bar, ALPHA)
            assert 0.3 < exact < 0.9
            assert abs(exact - heuristic) < 0.01

    def test_monotone_in_n(self):
        powers = [
            tost_power_exact(0.0, MARGIN, n, 0.005, 2e-5, ALPHA)
            for n in (2, 3, 5, 8, 12, 20)
        ]
        assert all(b > a for a, b in zip(powers, powers[1:]))

    def test_seeds_needed_exact_round_trip(self):
        n_star = seeds_needed(
            margin=MARGIN, power=0.8, sigma_seed=0.006, v_bar=3e-5, alpha=ALPHA,
            exact=True,
        )
        assert tost_power_exact(0.0, MARGIN, n_star, 0.006, 3e-5, ALPHA) >= 0.8
        assert tost_power_exact(0.0, MARGIN, n_star - 1, 0.006, 3e-5, ALPHA) < 0.8


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
