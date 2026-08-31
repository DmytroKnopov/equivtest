"""sim_power.py — empirical vs analytic TOST power over the full grid.

Grid: sigma_seed x n x theta x Delta, margin 0.01, alpha 0.05,
2000 replications per cell, m0 = m1 = 10 000 (CIFAR-10-test scale); Delta
includes both boundaries of the composite null (+-margin). Runs are
simulated with independent per-run test scores (image_rho = 0), the regime in
which the per-run AUROC variance is exactly sigma_seed^2 + v and the power
formula applies as stated; the shared-test-set correction is a case-study
matter (see case_study.py).

Output: results/sim_power.json (one record per cell: empirical power
with MC error, analytic power in t- and normal flavours, the v_bar used).

Usage (from repo root):
    python experiments/sim_power.py [--smoke]
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
from equivtest.dgp import simulate_runs
from equivtest.power import tost_power
from equivtest.tost import welch_tost

MARGIN = 0.01
ALPHA = 0.05
M0 = M1 = 10_000
N_REP = 2000
ROOT_SEED = 20260709
N_WORKERS = 12

SIGMA_SEED_GRID = [0.002, 0.005, 0.01]
N_GRID = [2, 3, 5, 8, 12, 20]
THETA_GRID = [0.55, 0.75, 0.90]
DELTA_GRID = [0.0, 0.005, 0.01, 0.02]
# Lower boundary of the composite null; seeded from a separate stream so the
# original cells keep their seeds (and results) unchanged.
DELTA_GRID_NEG = [-0.01]

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def estimate_v_bar(theta: float, m0: int, m1: int, entropy: int, n_runs: int = 200) -> float:
    """Mean DeLong variance of a single run at level theta (sigma_seed = 0)."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    runs = simulate_runs(theta, 0.0, n_runs, m0, m1, rng)
    return float(np.mean([delong_variance(r.id_scores, r.ood_scores)[0] for r in runs]))


def run_cell(args: tuple) -> dict:
    sigma_seed, n, theta, delta, n_rep, seed_entropy, v_bar = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    t0 = time.perf_counter()
    n_equivalent = 0
    for _ in range(n_rep):
        runs_a = simulate_runs(theta + delta, sigma_seed, n, M0, M1, rng)
        runs_b = simulate_runs(theta, sigma_seed, n, M0, M1, rng)
        theta_a = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_a])
        theta_b = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_b])
        try:
            res = welch_tost(theta_a, theta_b, margin=MARGIN, alpha=ALPHA)
        except ValueError:  # zero between-seed variance (possible at n = 2)
            continue
        n_equivalent += int(res.equivalent)
    p_emp = n_equivalent / n_rep
    return {
        "sigma_seed": sigma_seed,
        "n": n,
        "theta": theta,
        "delta": delta,
        "margin": MARGIN,
        "alpha": ALPHA,
        "m0": M0,
        "m1": M1,
        "n_rep": n_rep,
        "v_bar": v_bar,
        "power_empirical": p_emp,
        "power_mc_se": float(np.sqrt(p_emp * (1 - p_emp) / n_rep)),
        "power_analytic_t": tost_power(delta, MARGIN, n, sigma_seed, v_bar, ALPHA),
        "power_analytic_normal": tost_power(
            delta, MARGIN, n, sigma_seed, v_bar, ALPHA, use_t=False
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny grid, 30 reps")
    args = parser.parse_args()

    if args.smoke:
        grid_sigma, grid_n = [0.005], [2, 5]
        grid_theta, grid_delta = [0.55], [0.0, 0.02]
        grid_delta_neg = []
        n_rep, out_name = 30, "sim_power_smoke.json"
    else:
        grid_sigma, grid_n = SIGMA_SEED_GRID, N_GRID
        grid_theta, grid_delta = THETA_GRID, DELTA_GRID
        grid_delta_neg = DELTA_GRID_NEG
        n_rep, out_name = N_REP, "sim_power.json"

    v_bar = {
        theta: estimate_v_bar(theta, M0, M1, ROOT_SEED + i)
        for i, theta in enumerate(grid_theta)
    }
    print(f"v_bar estimates: { {t: f'{v:.3e}' for t, v in v_bar.items()} }", flush=True)

    cells = [
        (s, n, t, d)
        for s in grid_sigma for n in grid_n for t in grid_theta for d in grid_delta
    ]
    seeds = np.random.SeedSequence(ROOT_SEED).generate_state(len(cells)).tolist()
    cells_neg = [
        (s, n, t, d)
        for s in grid_sigma for n in grid_n for t in grid_theta for d in grid_delta_neg
    ]
    seeds_neg = (
        np.random.SeedSequence((ROOT_SEED, 1)).generate_state(len(cells_neg)).tolist()
        if cells_neg else []
    )
    tasks = [
        (s, n, t, d, n_rep, seed, v_bar[t])
        for (s, n, t, d), seed in zip(cells + cells_neg, seeds + seeds_neg)
    ]

    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(run_cell, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {time.perf_counter()-t0:.0f}s", flush=True)

    results.sort(key=lambda r: (r["sigma_seed"], r["theta"], r["delta"], r["n"]))
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / out_name
    payload = {
        "config": {
            "margin": MARGIN, "alpha": ALPHA, "m0": M0, "m1": M1,
            "n_rep": n_rep, "root_seed": ROOT_SEED, "image_rho": 0.0,
            "v_bar_per_theta": v_bar,
        },
        "cells": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} cells, {time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
