"""Tests for equivtest.accuracy: per-run accuracy and its two-level summaries.

Plan:
  (a) accuracy is the mean of the 0/1 vector; binomial_variance is p(1-p)/m;
  (b) between_run_correlation matches the mean of the off-diagonal entries of
      np.corrcoef on small matrices, is ~0 for independent runs and 1 for
      identical runs, and drops constant rows;
  (c) item_agreement shares add to one and match a hand-built matrix.
"""
import numpy as np
import pytest

from equivtest.accuracy import (
    accuracy,
    between_run_correlation,
    binomial_variance,
    item_agreement,
)


class TestAccuracyAndBinomialVariance:
    def test_accuracy_is_the_mean(self):
        assert accuracy(np.array([1, 0, 1, 1])) == pytest.approx(0.75)
        assert accuracy(np.ones(10, dtype=np.uint8)) == 1.0

    def test_accuracy_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            accuracy(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            accuracy(np.zeros(0))

    @pytest.mark.parametrize("p,m", [(0.5, 100), (0.9, 10_000), (0.0, 7), (1.0, 7)])
    def test_binomial_variance_value(self, p, m):
        assert binomial_variance(p, m) == pytest.approx(p * (1 - p) / m, abs=1e-15)

    def test_binomial_variance_matches_empirical_variance_of_a_mean(self):
        rng = np.random.default_rng(0)
        p, m = 0.7, 500
        means = rng.binomial(1, p, size=(20_000, m)).mean(axis=1)
        assert means.var() == pytest.approx(binomial_variance(p, m), rel=0.05)

    def test_binomial_variance_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            binomial_variance(0.5, 0)
        with pytest.raises(ValueError):
            binomial_variance(1.5, 10)


class TestBetweenRunCorrelation:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_corrcoef_on_small_matrices(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, size=(6, 40)).astype(np.uint8)
        c = np.corrcoef(x.astype(float))
        off = c[np.triu_indices(6, k=1)]
        assert between_run_correlation(x) == pytest.approx(off.mean(), abs=1e-12)

    def test_independent_runs_have_correlation_near_zero(self):
        rng = np.random.default_rng(3)
        x = rng.binomial(1, 0.6, size=(50, 5_000)).astype(np.uint8)
        # Pairwise correlations have sd ~ 1/sqrt(m); the mean over 1225 pairs is tighter.
        assert abs(between_run_correlation(x)) < 0.01

    def test_identical_runs_have_correlation_one(self):
        rng = np.random.default_rng(4)
        row = rng.integers(0, 2, size=300).astype(np.uint8)
        x = np.tile(row, (7, 1))
        assert between_run_correlation(x) == pytest.approx(1.0, abs=1e-12)

    def test_shared_item_difficulty_induces_positive_correlation(self):
        rng = np.random.default_rng(5)
        m = 4_000
        difficulty = rng.uniform(0.1, 0.9, size=m)
        x = (rng.uniform(size=(30, m)) < difficulty).astype(np.uint8)
        rho = between_run_correlation(x)
        # Var(p_i) / (Var(p_i) + E[p_i(1-p_i)]) for p_i ~ U(0.1, 0.9):
        # 0.0533 / 0.25 = 0.213.
        assert rho == pytest.approx(0.213, abs=0.02)

    def test_constant_rows_are_dropped(self):
        rng = np.random.default_rng(6)
        x = rng.integers(0, 2, size=(4, 50)).astype(np.uint8)
        with_const = np.vstack([x, np.ones((1, 50), dtype=np.uint8)])
        assert between_run_correlation(with_const) == pytest.approx(
            between_run_correlation(x), abs=1e-12
        )
        assert np.isnan(between_run_correlation(np.ones((3, 10), dtype=np.uint8)))

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError):
            between_run_correlation(np.zeros(10))


class TestItemAgreement:
    def test_hand_built_matrix(self):
        x = np.array([[1, 0, 1, 1],
                      [1, 0, 0, 1],
                      [1, 0, 1, 1]], dtype=np.uint8)
        shares = item_agreement(x)
        assert shares["always_correct"] == pytest.approx(0.5)
        assert shares["never_correct"] == pytest.approx(0.25)
        assert shares["mixed"] == pytest.approx(0.25)

    def test_shares_sum_to_one(self):
        rng = np.random.default_rng(7)
        x = rng.integers(0, 2, size=(5, 200)).astype(np.uint8)
        shares = item_agreement(x)
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
