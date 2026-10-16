"""sim_power_robust.py — power approximation under non-Gaussian seed effects.

Robustness companion to sim_power.py: the shifted-t power formula assumes
Gaussian seed effects; here the seed effects a_s are drawn from three
standardised laws (mean 0, variance sigma_seed^2) and the empirical TOST
rejection rate is compared with the same analytic formula:

  normal   -- reference, duplicates the sim_power.py regime;
  skewexp  -- centred exponential (skewness 2), a one-sided seed law;
  t3       -- Student t with 3 df scaled to unit variance (heavy tails).

Grid: dist x sigma_seed x n x Delta at theta = 0.75, margin 0.01, alpha 0.05,
independent per-run test scores, 2000 replications per cell. Delta = +margin
and -margin rows measure the realised size at the two boundaries of the
composite null.

Output: results/sim_power_robust.json.

Usage (from repo root):
    python experiments/sim_power_robust.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equivtest.auroc import auroc, delong_variance
from equivtest.dgp import Run, shift_from_theta, simulate_runs
from equivtest.power import tost_power
from equivtest.tost import welch_tost

MARGIN = 0.01
ALPHA = 0.05
THETA = 0.75
M0 = M1 = 10_000
N_REP = 2000
ROOT_SEED = 20260711
N_WORKERS = 12

DIST_GRID = ["normal", "skewexp", "t3"]
SIGMA_SEED_GRID = [0.005, 0.01]
N_GRID = [2, 3, 5, 12]
DELTA_GRID = [0.0, 0.005, 0.01]
# Lower boundary of the composite null; seeded from a separate stream so the
# original cells keep their seeds (and results) unchanged.
DELTA_GRID_NEG = [-0.01]

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def seed_effects(dist: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Standardised (mean 0, variance 1) seed-effect draws."""
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "skewexp":
        return rng.exponential(1.0, size) - 1.0
    if dist == "t3":
        return rng.standard_t(3, size) / np.sqrt(3.0)
    raise ValueError(f"unknown dist {dist!r}")


def simulate_runs_dist(
    mu: float, sigma_seed: float, dist: str, n_seeds: int,
    m0: int, m1: int, rng: np.random.Generator,
) -> list[Run]:
    """simulate_runs with a configurable seed-effect law.

    theta_s is clipped to (0.505, 0.999) to keep the binormal link defined;
    the clip probability peaks at about 3e-5 per draw (t3 at sigma = 0.01),
    about 11 expected clips over the whole grid; the t3 tails carry variance
    far out, so clipping lowers the t3 variance by 3.4% at sigma = 0.01 and
    1.7% at sigma = 0.005 (factors 0.966 and 0.983). sigma is therefore the
    pre-clipping scale, and the formula, which uses it, errs conservatively.
    """
    effects = seed_effects(dist, n_seeds, rng)
    runs = []
    for a in effects:
        theta_s = float(np.clip(mu + sigma_seed * a, 0.505, 0.999))
        shift = shift_from_theta(theta_s)
        runs.append(Run(rng.standard_normal(m0), shift + rng.standard_normal(m1), theta_s))
    return runs


def estimate_v_bar(theta: float, m0: int, m1: int, entropy: int, n_runs: int = 200) -> float:
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    runs = simulate_runs(theta, 0.0, n_runs, m0, m1, rng)
    return float(np.mean([delong_variance(r.id_scores, r.ood_scores)[0] for r in runs]))


def run_cell(args: tuple) -> dict:
    dist, sigma_seed, n, delta, n_rep, seed_entropy, v_bar = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    n_equivalent = 0
    n_skipped = 0
    for _ in range(n_rep):
        runs_a = simulate_runs_dist(THETA + delta, sigma_seed, dist, n, M0, M1, rng)
        runs_b = simulate_runs_dist(THETA, sigma_seed, dist, n, M0, M1, rng)
        theta_a = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_a])
        theta_b = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_b])
        try:
            res = welch_tost(theta_a, theta_b, margin=MARGIN, alpha=ALPHA)
        except ValueError:
            n_skipped += 1
            continue
        n_equivalent += int(res.equivalent)
    p_emp = n_equivalent / n_rep
    return {
        "dist": dist,
        "sigma_seed": sigma_seed,
        "n": n,
        "theta": THETA,
        "delta": delta,
        "margin": MARGIN,
        "alpha": ALPHA,
        "m0": M0,
        "m1": M1,
        "n_rep": n_rep,
        "n_skipped": n_skipped,
        "v_bar": v_bar,
        "power_empirical": p_emp,
        "power_mc_se": float(np.sqrt(p_emp * (1 - p_emp) / n_rep)),
        "power_analytic_t": tost_power(delta, MARGIN, n, sigma_seed, v_bar, ALPHA),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny grid, 30 reps")
    args = parser.parse_args()

    if args.smoke:
        grid_dist, grid_sigma = ["skewexp"], [0.01]
        grid_n, grid_delta = [2, 5], [0.0]
        grid_delta_neg = []
        n_rep, out_name = 30, "sim_power_robust_smoke.json"
    else:
        grid_dist, grid_sigma = DIST_GRID, SIGMA_SEED_GRID
        grid_n, grid_delta = N_GRID, DELTA_GRID
        grid_delta_neg = DELTA_GRID_NEG
        n_rep, out_name = N_REP, "sim_power_robust.json"

    v_bar = estimate_v_bar(THETA, M0, M1, ROOT_SEED)
    print(f"v_bar estimate at theta={THETA}: {v_bar:.3e}", flush=True)

    cells = [
        (dist, s, n, d)
        for dist in grid_dist for s in grid_sigma for n in grid_n for d in grid_delta
    ]
    seeds = np.random.SeedSequence(ROOT_SEED).generate_state(len(cells)).tolist()
    cells_neg = [
        (dist, s, n, d)
        for dist in grid_dist for s in grid_sigma for n in grid_n for d in grid_delta_neg
    ]
    seeds_neg = (
        np.random.SeedSequence((ROOT_SEED, 1)).generate_state(len(cells_neg)).tolist()
        if cells_neg else []
    )
    tasks = [
        (dist, s, n, d, n_rep, seed, v_bar)
        for (dist, s, n, d), seed in zip(cells + cells_neg, seeds + seeds_neg)
    ]

    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(run_cell, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {time.perf_counter()-t0:.0f}s", flush=True)

    results.sort(key=lambda r: (r["dist"], r["sigma_seed"], r["delta"], r["n"]))
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / out_name
    payload = {
        "config": {
            "margin": MARGIN, "alpha": ALPHA, "theta": THETA, "m0": M0, "m1": M1,
            "n_rep": n_rep, "root_seed": ROOT_SEED, "image_rho": 0.0,
            "v_bar": v_bar, "dists": grid_dist,
        },
        "cells": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} cells, {time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
