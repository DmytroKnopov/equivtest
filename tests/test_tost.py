"""Tests for equivtest.tost: seed-level Welch-TOST.

Contents:
  - REGRESSION: reproduce the published case-study numbers (TOST p = 0.33,
    df ~= 1.2) on the macro-5 AUROCs of the four training runs, checked
    against data/hier_bootstrap_tost.json, and recompute those macro-5
    values from the raw per-image scores with equivtest.auroc;
  - synthetic sanity: clear equivalence => p < 0.05, clear non-equivalence
    (delta = 2 * margin) => p ~= 1;
  - TOST <-> (1 - 2 alpha) CI duality.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from equivtest.auroc import auroc
from equivtest.tost import TostResult, welch_tost

REPO = Path(__file__).resolve().parents[1]
PUBLISHED_JSON = REPO / "data" / "hier_bootstrap_tost.json"
SCORES_DIR = REPO / "data" / "ood_scores"

# The four case-study runs and the OOD suite.
SGLD_STEMS = ["sgld_seed42", "sgld_seed123"]
PC_STEMS = ["pc_seed42", "pc_seed123"]
DATASETS = ["svhn", "cifar100", "dtd", "lsun", "isun"]


def _npz_path(stem: str, ds: str) -> Path:
    return SCORES_DIR / f"{stem}__{ds}.npz"


def _macro5(stem: str) -> float:
    with np.load(_npz_path(stem, "cifar10")) as f:
        id_scores = f["scores"].astype(np.float64)
    per_ds = []
    for ds in DATASETS:
        with np.load(_npz_path(stem, ds)) as f:
            per_ds.append(auroc(id_scores, f["scores"].astype(np.float64)))
    return float(np.mean(per_ds))


@pytest.fixture(scope="module")
def published():
    with open(PUBLISHED_JSON, encoding="utf-8") as f:
        return json.load(f)


class TestRegressionArticle1:
    """The port must reproduce the published numbers exactly; drift = bug."""

    def test_reproduces_published_tost(self, published):
        sgld = published["point_estimates"]["sgld_macro5"]
        pc = published["point_estimates"]["pc_macro5"]
        ref = published["tost"]
        # Convention: delta = mean(theta_a) - mean(theta_b); the case study
        # reported delta = PC - SGLD, so PC goes first.
        res = welch_tost(np.array(pc), np.array(sgld), margin=ref["margin"])
        assert res.delta == pytest.approx(ref["diff"], abs=1e-12)
        assert res.se == pytest.approx(ref["se"], abs=1e-12)
        assert res.df == pytest.approx(ref["df"], abs=1e-9)
        assert res.p_lower == pytest.approx(ref["p_low"], abs=1e-9)
        assert res.p_upper == pytest.approx(ref["p_high"], abs=1e-9)
        assert res.p_tost == pytest.approx(ref["p_tost"], abs=1e-9)
        assert res.equivalent is False
        # The published headline numbers.
        assert round(res.p_tost, 2) == 0.33
        assert res.df == pytest.approx(1.21, abs=0.005)

    @pytest.mark.skipif(
        not SCORES_DIR.exists(), reason="raw case-study scores not on this machine"
    )
    def test_macro5_recomputed_from_raw_scores(self, published):
        sgld = [_macro5(s) for s in SGLD_STEMS]
        pc = [_macro5(s) for s in PC_STEMS]
        assert sgld == pytest.approx(
            published["point_estimates"]["sgld_macro5"], abs=1e-12
        )
        assert pc == pytest.approx(published["point_estimates"]["pc_macro5"], abs=1e-12)


class TestSyntheticBehaviour:
    def test_clear_equivalence_rejected_null(self):
        rng = np.random.default_rng(0)
        theta_a = 0.60 + 1e-4 * rng.standard_normal(50)
        theta_b = 0.60 + 1e-4 * rng.standard_normal(50)
        res = welch_tost(theta_a, theta_b, margin=0.01)
        assert res.p_tost < 0.05
        assert res.equivalent is True

    def test_clear_non_equivalence_keeps_null(self):
        rng = np.random.default_rng(1)
        margin = 0.01
        theta_a = 0.60 + 2 * margin + 1e-4 * rng.standard_normal(50)
        theta_b = 0.60 + 1e-4 * rng.standard_normal(50)
        res = welch_tost(theta_a, theta_b, margin=margin)
        assert res.p_tost > 0.9
        assert res.equivalent is False

    def test_tost_ci_duality(self):
        # Equivalence at level alpha <=> the (1 - 2 alpha) CI lies inside
        # (-margin, +margin) -- checked over a spread of random designs.
        rng = np.random.default_rng(2)
        outcomes = set()
        for _ in range(200):
            n_a = int(rng.integers(2, 10))
            n_b = int(rng.integers(2, 10))
            delta = float(rng.uniform(-0.03, 0.03))
            sigma = float(rng.uniform(0.001, 0.02))
            theta_a = 0.6 + delta + sigma * rng.standard_normal(n_a)
            theta_b = 0.6 + sigma * rng.standard_normal(n_b)
            res = welch_tost(theta_a, theta_b, margin=0.01, alpha=0.05)
            lo, hi = res.ci90
            inside = (-res.margin < lo) and (hi < res.margin)
            assert res.equivalent == inside
            assert lo <= res.delta <= hi
            outcomes.add(res.equivalent)
        assert outcomes == {True, False}  # both branches actually exercised

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError):
            welch_tost(np.array([0.6, 0.6]), np.array([0.6, 0.6]), margin=0.01)

    def test_result_type(self):
        res = welch_tost(np.array([0.60, 0.61]), np.array([0.605, 0.612]), margin=0.01)
        assert isinstance(res, TostResult)
        assert res.p_tost == max(res.p_lower, res.p_upper)
        assert res.n_a == 2 and res.n_b == 2
