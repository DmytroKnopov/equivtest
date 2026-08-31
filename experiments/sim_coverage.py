"""sim_coverage.py — coverage of the two-level CI vs the naive per-image CI.

Setting: both methods equal (true Delta = 0), all
runs score the SAME test set with cross-run score correlation image_rho = 0.8
(the value measured on the case-study data: 0.79-0.92 across OOD sets). The
naive baseline resamples images only, with all runs fixed -- the analogue of
the per-image paired bootstrap of the published case study -- and ignores the between-seed
level entirely; the two-level CI is equivtest.hierboot.hierarchical_ci.

Output: results/sim_coverage.json (per cell: empirical coverage and
mean CI width for both interval types).

Usage (from repo root):
    python experiments/sim_coverage.py [--smoke]
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

from equivtest.auroc import auroc
from equivtest.dgp import simulate_experiment
from equivtest.hierboot import hierarchical_ci

THETA = 0.6
IMAGE_RHO = 0.8
M0 = M1 = 2000
ALPHA = 0.05
B = 400
N_REP = 1000
CHUNK = 50
ROOT_SEED = 20260710
N_WORKERS = 12

SIGMA_SEED_GRID = [0.002, 0.005, 0.01]
N_GRID = [2, 5]

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def naive_image_ci(
    runs_a: list, runs_b: list, B: int, alpha: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Images-only paired bootstrap with all runs fixed (no seed level)."""
    m0 = len(runs_a[0].id_scores)
    m1 = len(runs_a[0].ood_scores)
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx_in = rng.integers(0, m0, size=m0)
        idx_out = rng.integers(0, m1, size=m1)
        stat_a = np.mean(
            [auroc(r.id_scores[idx_in], r.ood_scores[idx_out]) for r in runs_a]
        )
        stat_b = np.mean(
            [auroc(r.id_scores[idx_in], r.ood_scores[idx_out]) for r in runs_b]
        )
        boot[b] = stat_a - stat_b
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def run_chunk(args: tuple) -> dict:
    sigma_seed, n, n_rep, b_boot, seed_entropy = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    hits_two, hits_naive = 0, 0
    width_two, width_naive = 0.0, 0.0
    for _ in range(n_rep):
        runs_a, runs_b = simulate_experiment(
            THETA, THETA, sigma_seed, n, M0, M1, rng, image_rho=IMAGE_RHO
        )
        lo, hi, _ = hierarchical_ci(runs_a, runs_b, B=b_boot, alpha=ALPHA, rng=rng)
        hits_two += int(lo <= 0.0 <= hi)
        width_two += hi - lo
        lo_n, hi_n = naive_image_ci(runs_a, runs_b, b_boot, ALPHA, rng)
        hits_naive += int(lo_n <= 0.0 <= hi_n)
        width_naive += hi_n - lo_n
    return {
        "sigma_seed": sigma_seed,
        "n": n,
        "n_rep": n_rep,
        "hits_two": hits_two,
        "hits_naive": hits_naive,
        "width_two_sum": width_two,
        "width_naive_sum": width_naive,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run")
    args = parser.parse_args()

    if args.smoke:
        grid_sigma, grid_n = [0.005], [2]
        n_rep, b_boot, chunk, out_name = 10, 50, 5, "sim_coverage_smoke.json"
    else:
        grid_sigma, grid_n = SIGMA_SEED_GRID, N_GRID
        n_rep, b_boot, chunk, out_name = N_REP, B, CHUNK, "sim_coverage.json"

    cells = [(s, n) for s in grid_sigma for n in grid_n]
    tasks = []
    seed_pool = iter(
        np.random.SeedSequence(ROOT_SEED).generate_state(
            len(cells) * (n_rep // chunk) + len(cells)
        ).tolist()
    )
    for s, n in cells:
        done = 0
        while done < n_rep:
            reps = min(chunk, n_rep - done)
            tasks.append((s, n, reps, b_boot, next(seed_pool)))
            done += reps

    agg: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(run_chunk, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            key = (r["sigma_seed"], r["n"])
            a = agg.setdefault(
                key,
                {"n_rep": 0, "hits_two": 0, "hits_naive": 0,
                 "width_two_sum": 0.0, "width_naive_sum": 0.0},
            )
            for f in a:
                a[f] += r[f]
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {time.perf_counter()-t0:.0f}s", flush=True)

    results = []
    for (s, n), a in sorted(agg.items()):
        cov_two = a["hits_two"] / a["n_rep"]
        cov_naive = a["hits_naive"] / a["n_rep"]
        results.append({
            "sigma_seed": s,
            "n": n,
            "n_rep": a["n_rep"],
            "coverage_two_level": cov_two,
            "coverage_naive": cov_naive,
            "coverage_mc_se_two": float(np.sqrt(cov_two * (1 - cov_two) / a["n_rep"])),
            "coverage_mc_se_naive": float(
                np.sqrt(cov_naive * (1 - cov_naive) / a["n_rep"])
            ),
            "mean_width_two_level": a["width_two_sum"] / a["n_rep"],
            "mean_width_naive": a["width_naive_sum"] / a["n_rep"],
        })

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / out_name
    payload = {
        "config": {
            "theta": THETA, "delta_true": 0.0, "image_rho": IMAGE_RHO,
            "m0": M0, "m1": M1, "alpha": ALPHA, "B": b_boot,
            "n_rep": n_rep, "root_seed": ROOT_SEED,
        },
        "cells": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} cells, {time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
