"""sim_coverage_ext.py — extended coverage map for the two-level bootstrap.

Extends sim_coverage.py (6 cells: sigma x n at rho = 0.8, theta = 0.6,
Delta = 0) along the axes the six-cell probe held fixed:

  block A  n in {2, 3, 5, 8, 12} x sigma in {0.002, 0.005, 0.01}
           (full n x sigma map at the baseline rho / theta / Delta);
  block B  rho in {0.5, 0.95}  x n in {2, 5} x sigma in {0.005, 0.01};
  block C  Delta_true = 0.005  x n in {2, 5} x sigma in {0.005, 0.01};
  block D  theta = 0.75        x n in {2, 5} x sigma in {0.005, 0.01}.

Everything else as in sim_coverage.py: shared test set, m0 = m1 = 2000,
B = 400 bootstrap iterations, 1000 replications per cell, coverage of the
true Delta by the nominal 95% interval, naive per-image bootstrap baseline.

Output: results/sim_coverage_ext.json.

Usage (from repo root):
    python experiments/sim_coverage_ext.py [--smoke]
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

from equivtest.dgp import simulate_experiment
from equivtest.hierboot import hierarchical_ci

from sim_coverage import naive_image_ci

M0 = M1 = 2000
ALPHA = 0.05
B = 400
N_REP = 1000
CHUNK = 50
ROOT_SEED = 20260711
N_WORKERS = 12

BASE_RHO, BASE_THETA = 0.8, 0.6

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def build_cells() -> list[dict]:
    cells = []
    for n in [2, 3, 5, 8, 12]:
        for s in [0.002, 0.005, 0.01]:
            cells.append({"block": "A", "sigma_seed": s, "n": n,
                          "rho": BASE_RHO, "theta": BASE_THETA, "delta_true": 0.0})
    for rho in [0.5, 0.95]:
        for n in [2, 5]:
            for s in [0.005, 0.01]:
                cells.append({"block": "B", "sigma_seed": s, "n": n,
                              "rho": rho, "theta": BASE_THETA, "delta_true": 0.0})
    for n in [2, 5]:
        for s in [0.005, 0.01]:
            cells.append({"block": "C", "sigma_seed": s, "n": n,
                          "rho": BASE_RHO, "theta": BASE_THETA, "delta_true": 0.005})
    for n in [2, 5]:
        for s in [0.005, 0.01]:
            cells.append({"block": "D", "sigma_seed": s, "n": n,
                          "rho": BASE_RHO, "theta": 0.75, "delta_true": 0.0})
    return cells


def run_chunk(args: tuple) -> dict:
    cell, n_rep, b_boot, seed_entropy = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    s, n = cell["sigma_seed"], cell["n"]
    rho, theta, delta_true = cell["rho"], cell["theta"], cell["delta_true"]
    hits_two, hits_naive = 0, 0
    width_two, width_naive = 0.0, 0.0
    for _ in range(n_rep):
        runs_a, runs_b = simulate_experiment(
            theta + delta_true, theta, s, n, M0, M1, rng, image_rho=rho
        )
        lo, hi, _ = hierarchical_ci(runs_a, runs_b, B=b_boot, alpha=ALPHA, rng=rng)
        hits_two += int(lo <= delta_true <= hi)
        width_two += hi - lo
        lo_n, hi_n = naive_image_ci(runs_a, runs_b, b_boot, ALPHA, rng)
        hits_naive += int(lo_n <= delta_true <= hi_n)
        width_naive += hi_n - lo_n
    return {
        "cell": cell,
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

    cells = build_cells()
    if args.smoke:
        cells = [cells[0], cells[15]]
        n_rep, b_boot, chunk, out_name = 10, 50, 5, "sim_coverage_ext_smoke.json"
    else:
        n_rep, b_boot, chunk, out_name = N_REP, B, CHUNK, "sim_coverage_ext.json"

    tasks = []
    seed_pool = iter(
        np.random.SeedSequence(ROOT_SEED).generate_state(
            len(cells) * (n_rep // chunk + 1)
        ).tolist()
    )
    for cell in cells:
        done = 0
        while done < n_rep:
            reps = min(chunk, n_rep - done)
            tasks.append((cell, reps, b_boot, next(seed_pool)))
            done += reps

    agg: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(run_chunk, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            c = r["cell"]
            key = (c["block"], c["sigma_seed"], c["n"], c["rho"],
                   c["theta"], c["delta_true"])
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
    for (block, s, n, rho, theta, delta_true), a in sorted(agg.items()):
        cov_two = a["hits_two"] / a["n_rep"]
        cov_naive = a["hits_naive"] / a["n_rep"]
        results.append({
            "block": block,
            "sigma_seed": s,
            "n": n,
            "rho": rho,
            "theta": theta,
            "delta_true": delta_true,
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
            "m0": M0, "m1": M1, "alpha": ALPHA, "B": b_boot,
            "n_rep": n_rep, "root_seed": ROOT_SEED,
            "base_rho": BASE_RHO, "base_theta": BASE_THETA,
        },
        "cells": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} cells, {time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
