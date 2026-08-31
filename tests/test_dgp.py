"""Tests for equivtest.dgp: binormal two-level (seed x image) data generator.

Contents:
  (a) mean empirical AUROC over 500 seeds ~= mu (within 3 SE);
  (b) sample variance of the true per-seed theta_s ~= sigma_seed^2;
  (c) point checks of the binormal link theta(m) = Phi(m / sqrt(2)).
"""
import numpy as np
import pytest
from scipy.stats import norm

from equivtest.auroc import auroc
from equivtest.dgp import (
    Run,
    shift_from_theta,
    simulate_experiment,
    simulate_runs,
    theta_from_shift,
)


class TestBinormalLink:
    def test_zero_shift_gives_chance_level(self):
        assert theta_from_shift(0.0) == pytest.approx(0.5, abs=1e-15)

    def test_known_point_value(self):
        # theta(m) = Phi(m / sqrt(2)): plug m = sqrt(2) * Phi^{-1}(0.9).
        shift = np.sqrt(2.0) * norm.ppf(0.9)
        assert theta_from_shift(shift) == pytest.approx(0.9, abs=1e-12)

    def test_round_trip(self):
        for theta in (0.55, 0.61, 0.75, 0.90):
            assert theta_from_shift(shift_from_theta(theta)) == pytest.approx(
                theta, abs=1e-12
            )

    def test_generator_realises_the_link_empirically(self):
        # sigma_seed = 0: every run has theta_true = mu exactly; with large
        # samples the empirical AUROC must converge to it.
        rng = np.random.default_rng(0)
        (run,) = simulate_runs(
            mu=0.8, sigma_seed=0.0, n_seeds=1, m0=100_000, m1=100_000, rng=rng
        )
        assert run.theta_true == pytest.approx(0.8, abs=1e-15)
        assert auroc(run.id_scores, run.ood_scores) == pytest.approx(0.8, abs=0.005)


class TestSimulateRuns:
    def test_shapes_and_types(self):
        rng = np.random.default_rng(1)
        runs = simulate_runs(mu=0.6, sigma_seed=0.005, n_seeds=7, m0=50, m1=80, rng=rng)
        assert len(runs) == 7
        for run in runs:
            assert isinstance(run, Run)
            assert run.id_scores.shape == (50,)
            assert run.ood_scores.shape == (80,)
            assert 0.0 < run.theta_true < 1.0

    def test_mean_empirical_auroc_matches_mu(self):
        rng = np.random.default_rng(2)
        mu, n_seeds = 0.6, 500
        runs = simulate_runs(
            mu=mu, sigma_seed=0.005, n_seeds=n_seeds, m0=2000, m1=2000, rng=rng
        )
        aurocs = np.array([auroc(r.id_scores, r.ood_scores) for r in runs])
        se = aurocs.std(ddof=1) / np.sqrt(n_seeds)
        assert aurocs.mean() == pytest.approx(mu, abs=3 * se)

    def test_theta_true_dispersion_matches_sigma_seed(self):
        rng = np.random.default_rng(3)
        mu, sigma_seed, n_seeds = 0.6, 0.005, 500
        runs = simulate_runs(
            mu=mu, sigma_seed=sigma_seed, n_seeds=n_seeds, m0=10, m1=10, rng=rng
        )
        thetas = np.array([r.theta_true for r in runs])
        assert thetas.mean() == pytest.approx(mu, abs=3 * sigma_seed / np.sqrt(n_seeds))
        assert thetas.var(ddof=1) == pytest.approx(sigma_seed**2, rel=0.2)


class TestRejectionAndValidation:
    """theta draws outside (0, 1) are rejected and redrawn; invalid or
    degenerate parameters raise instead of looping forever."""

    def test_extreme_parameters_keep_theta_in_domain(self):
        # mu near 1 with a large spread forces frequent rejections; without
        # the redraw loop such draws leave (0, 1) and give an infinite
        # binormal shift.
        rng = np.random.default_rng(4)
        runs = simulate_runs(
            mu=0.99, sigma_seed=0.3, n_seeds=100, m0=5, m1=5, rng=rng
        )
        for run in runs:
            assert 0.0 < run.theta_true < 1.0
            assert np.isfinite(run.ood_scores).all()

    def test_redraw_consumes_the_documented_rng_stream(self):
        # Replay the accept/reject stream by hand: the generator must draw
        # one normal per attempt and nothing else between attempts.
        mu, sigma_seed, m0, m1 = 0.99, 0.3, 4, 6
        rng = np.random.default_rng(4)
        runs = simulate_runs(
            mu=mu, sigma_seed=sigma_seed, n_seeds=3, m0=m0, m1=m1, rng=rng
        )
        replay = np.random.default_rng(4)
        rejections = 0
        for run in runs:
            theta = mu + sigma_seed * replay.standard_normal()
            while not 0.0 < theta < 1.0:
                rejections += 1
                theta = mu + sigma_seed * replay.standard_normal()
            assert run.theta_true == theta
            np.testing.assert_array_equal(run.id_scores, replay.standard_normal(m0))
            np.testing.assert_array_equal(
                run.ood_scores, shift_from_theta(theta) + replay.standard_normal(m1)
            )
        assert rejections >= 1  # the branch under test actually fired

    def test_invalid_mu_raises(self):
        rng = np.random.default_rng(0)
        for mu in (0.0, 1.0, -0.2, 1.7, float("nan")):
            with pytest.raises(ValueError):
                simulate_runs(mu=mu, sigma_seed=0.01, n_seeds=1, m0=5, m1=5, rng=rng)
            with pytest.raises(ValueError):
                simulate_experiment(
                    mu_a=0.7, mu_b=mu, sigma_seed=0.01, n_seeds=1, m0=5, m1=5, rng=rng
                )

    def test_invalid_sigma_raises(self):
        rng = np.random.default_rng(0)
        for sigma in (-1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                simulate_runs(mu=0.7, sigma_seed=sigma, n_seeds=1, m0=5, m1=5, rng=rng)
            with pytest.raises(ValueError):
                simulate_experiment(
                    mu_a=0.7, mu_b=0.7, sigma_seed=sigma, n_seeds=1, m0=5, m1=5, rng=rng
                )

    def test_hopeless_acceptance_raises_instead_of_hanging(self):
        # Valid parameters but an absurd scale: acceptance probability per
        # draw ~ 4e-13, so the attempt cap must trip, not hang.
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError):
            simulate_runs(mu=0.5, sigma_seed=1e12, n_seeds=1, m0=5, m1=5, rng=rng)


class TestSharedImageEffects:
    """simulate_experiment: all runs of both methods score the SAME test
    images; image_rho is the cross-run correlation of same-image scores
    (measured at ~0.79-0.92 in the case-study data)."""

    def test_shapes_and_theta_levels(self):
        rng = np.random.default_rng(0)
        runs_a, runs_b = simulate_experiment(
            mu_a=0.60, mu_b=0.61, sigma_seed=0.005, n_seeds=4,
            m0=100, m1=150, rng=rng, image_rho=0.8,
        )
        assert len(runs_a) == 4 and len(runs_b) == 4
        for run in runs_a + runs_b:
            assert isinstance(run, Run)
            assert run.id_scores.shape == (100,)
            assert run.ood_scores.shape == (150,)
        assert np.mean([r.theta_true for r in runs_a]) == pytest.approx(0.60, abs=0.01)
        assert np.mean([r.theta_true for r in runs_b]) == pytest.approx(0.61, abs=0.01)

    def test_cross_run_correlation_matches_rho(self):
        rng = np.random.default_rng(1)
        m = 200_000
        runs_a, runs_b = simulate_experiment(
            mu_a=0.6, mu_b=0.6, sigma_seed=0.0, n_seeds=2,
            m0=m, m1=m, rng=rng, image_rho=0.8,
        )
        # Within a method, across methods, and on both score types.
        pairs = [
            (runs_a[0].id_scores, runs_a[1].id_scores),
            (runs_a[0].id_scores, runs_b[0].id_scores),
            (runs_a[0].ood_scores, runs_b[1].ood_scores),
        ]
        for x, y in pairs:
            assert np.corrcoef(x, y)[0, 1] == pytest.approx(0.8, abs=0.01)

    def test_zero_rho_gives_independent_runs(self):
        rng = np.random.default_rng(2)
        m = 100_000
        runs_a, runs_b = simulate_experiment(
            mu_a=0.6, mu_b=0.6, sigma_seed=0.0, n_seeds=2,
            m0=m, m1=m, rng=rng, image_rho=0.0,
        )
        assert np.corrcoef(runs_a[0].id_scores, runs_a[1].id_scores)[0, 1] == (
            pytest.approx(0.0, abs=0.02)
        )

    def test_marginals_and_link_preserved(self):
        # Shared effects must not distort the marginal N(0,1)/N(shift,1)
        # laws, hence the binormal link and the AUROC level stay intact.
        rng = np.random.default_rng(3)
        m = 100_000
        (run,), _ = simulate_experiment(
            mu_a=0.8, mu_b=0.8, sigma_seed=0.0, n_seeds=1,
            m0=m, m1=m, rng=rng, image_rho=0.8,
        )
        assert run.id_scores.mean() == pytest.approx(0.0, abs=0.02)
        assert run.id_scores.std() == pytest.approx(1.0, abs=0.02)
        assert run.ood_scores.std() == pytest.approx(1.0, abs=0.02)
        assert auroc(run.id_scores, run.ood_scores) == pytest.approx(0.8, abs=0.01)
