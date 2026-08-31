"""Tests for equivtest.auroc: Mann-Whitney AUROC + DeLong variance components.

Contents:
  (a) agreement with sklearn.metrics.roc_auc_score on random data, atol 1e-12,
      including tied scores (ties must be handled via average ranks);
  (b) small hand-computed examples with known answers;
  (c) DeLong variance on binormal data: shrinks with sample size and matches
      the empirical variance of AUROC over 2000 replications (rtol 0.1).
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from equivtest.auroc import auroc, delong_variance


def _sklearn_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Reference value: AUROC with OOD as the positive class (higher score = more OOD)."""
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))


class TestAurocAgainstSklearn:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("m0,m1", [(50, 50), (100, 37), (13, 211)])
    def test_matches_sklearn_on_continuous_data(self, seed, m0, m1):
        rng = np.random.default_rng(seed)
        id_scores = rng.normal(0.0, 1.0, size=m0)
        ood_scores = rng.normal(0.7, 1.3, size=m1)
        assert auroc(id_scores, ood_scores) == pytest.approx(
            _sklearn_auroc(id_scores, ood_scores), abs=1e-12
        )

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_matches_sklearn_with_heavy_ties(self, seed):
        rng = np.random.default_rng(seed)
        # Integer-quantised scores force many ties across and within classes.
        id_scores = rng.integers(0, 5, size=80).astype(np.float64)
        ood_scores = rng.integers(2, 7, size=60).astype(np.float64)
        assert auroc(id_scores, ood_scores) == pytest.approx(
            _sklearn_auroc(id_scores, ood_scores), abs=1e-12
        )


class TestAurocManualExamples:
    def test_three_by_three_no_ties(self):
        # Pairs (ood > id): 1.5 beats {0,1}; 2.5 beats {0,1,2}; 3.0 beats {0,1,2}
        # => (2 + 3 + 3) / 9 = 8/9.
        id_scores = np.array([0.0, 1.0, 2.0])
        ood_scores = np.array([1.5, 2.5, 3.0])
        assert auroc(id_scores, ood_scores) == pytest.approx(8.0 / 9.0, abs=1e-15)

    def test_three_by_three_with_ties(self):
        # ood=1: 1 win + 1 tie (0.5) = 1.5; ood=2: 2 wins + tie = 2.5; ood=3: 3.
        # => (1.5 + 2.5 + 3) / 9 = 7/9.
        id_scores = np.array([0.0, 1.0, 2.0])
        ood_scores = np.array([1.0, 2.0, 3.0])
        assert auroc(id_scores, ood_scores) == pytest.approx(7.0 / 9.0, abs=1e-15)

    def test_perfect_separation(self):
        assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0

    def test_identical_distributions_of_one_value(self):
        # All scores equal: every pair is a tie => AUROC = 0.5.
        assert auroc(np.zeros(4), np.zeros(6)) == pytest.approx(0.5, abs=1e-15)


class TestDelongVariance:
    def test_components_nonnegative_and_consistent(self):
        rng = np.random.default_rng(0)
        id_scores = rng.normal(0.0, 1.0, size=200)
        ood_scores = rng.normal(1.0, 1.0, size=300)
        v, s10, s01 = delong_variance(id_scores, ood_scores)
        assert s10 >= 0.0 and s01 >= 0.0
        assert v == pytest.approx(s10 / len(ood_scores) + s01 / len(id_scores), rel=1e-12)

    def test_variance_shrinks_with_sample_size(self):
        rng = np.random.default_rng(1)
        small_id = rng.normal(0.0, 1.0, size=100)
        small_ood = rng.normal(1.0, 1.0, size=100)
        big_id = rng.normal(0.0, 1.0, size=10_000)
        big_ood = rng.normal(1.0, 1.0, size=10_000)
        v_small, _, _ = delong_variance(small_id, small_ood)
        v_big, _, _ = delong_variance(big_id, big_ood)
        assert v_big < v_small / 10.0  # ~1/m scaling, generous slack

    def test_matches_empirical_variance_binormal(self):
        # Binormal model: ID ~ N(0,1), OOD ~ N(mu,1) => theta = Phi(mu/sqrt(2)).
        # Compare mean DeLong estimate with the empirical variance of AUROC
        # across 2000 independent replications (tolerance: rtol 0.1).
        rng = np.random.default_rng(2)
        mu, m0, m1, n_rep = 1.0, 400, 300, 2000
        aurocs = np.empty(n_rep)
        v_estimates = np.empty(n_rep)
        for r in range(n_rep):
            id_scores = rng.normal(0.0, 1.0, size=m0)
            ood_scores = rng.normal(mu, 1.0, size=m1)
            aurocs[r] = auroc(id_scores, ood_scores)
            v_estimates[r], _, _ = delong_variance(id_scores, ood_scores)
        empirical_var = aurocs.var(ddof=1)
        assert v_estimates.mean() == pytest.approx(empirical_var, rel=0.1)
