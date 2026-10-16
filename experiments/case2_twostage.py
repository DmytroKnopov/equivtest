"""case2_twostage.py -- the two-stage design run end to end on the seed zoo.

Second case study of article 2, complement to Analysis B of case2_modelzoo.py.
The paper recommends a pilot of a few runs per method, the design formula
n*(delta, power, s_pilot), then the confirmatory experiment. Analysis C reports
the pilot-to-pilot distribution of n*; this script measures what that
procedure actually delivers when it is carried through: the certification
rate of the final TOST when the confirmatory runs are fresh draws of the
size the pilot prescribed.

Data: the 1000 seed-zoo accuracies of results/case2_runs_modelzoo.csv
(one configuration, so Delta = 0 by construction and every certification is a
correct one; the achieved rate is the achieved power).

Per replication and margin:
  1. pilot: 2 n0 distinct runs, split into two arms of n0;
  2. design: n* = seeds_needed(delta, power, s_rule, v_bar = 0) with
     s_rule either the pooled pilot s (rule "plugin") or the 80% upper
     confidence bound of the pooled pilot variance on 2 (n0 - 1) degrees of
     freedom (rule "upper80"); recommendations below the floor of five are
     raised to five;
  3. confirmatory: 2 n* distinct runs disjoint from the pilot, split into two
     arms, Welch-TOST at delta; the certification is recorded together with
     n* and the total run count 2 n0 + 2 n*.
Reference points: the oracle design that uses the full-pool s (fresh arms of
that fixed size) and the same procedure under a Gaussian seed law with the
full-pool s (no finite pool), which isolates the sampling variability of the
pilot variance from the shape of the real seed law.

Usage (from repo root):
    python experiments_article2/case2_twostage.py [--reps 5000] [--gauss-reps 20000]
Outputs: results/case2_twostage.json and case2_twostage.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case2_common import ALPHA, ROOT_SEED, mc_se, tost_decision  # noqa: E402
from equivtest.power import seeds_needed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results"
RUNS_CSV = TABLES / "case2_runs_modelzoo.csv"

MARGINS = [0.01, 0.0173]
N0 = 5
POWER = 0.8
FLOOR = 5
UPPER_LEVEL = 0.8
RULES = ("plugin", "upper80")


def load_seed_zoo() -> np.ndarray:
    acc = []
    with open(RUNS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["zoo"] == "seed":
                acc.append(float(row["recomputed_accuracy"]))
    return np.asarray(acc, dtype=np.float64)


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))


def design_size(s_pilot: float, margin: float, rule: str, n0: int) -> tuple[int, int]:
    """(raw formula value, floored value) for the given sizing rule."""
    nu = 2 * (n0 - 1)
    if rule == "plugin":
        s_rule = s_pilot
    elif rule == "upper80":
        s_rule = s_pilot * float(np.sqrt(nu / chi2.ppf(1.0 - UPPER_LEVEL, df=nu)))
    else:
        raise ValueError(rule)
    raw = int(seeds_needed(margin, POWER, s_rule, 0.0, ALPHA)) if s_rule > 0 else 2
    return raw, max(raw, FLOOR)


def two_stage_finite(acc: np.ndarray, margin: float, rule: str, reps: int,
                     rng: np.random.Generator) -> dict:
    r = len(acc)
    hits = 0
    n_star = np.empty(reps, dtype=np.int64)
    raw_vals = np.empty(reps, dtype=np.int64)
    infeasible = 0
    for i in range(reps):
        pilot = rng.choice(r, size=2 * N0, replace=False)
        a0, b0 = acc[pilot[:N0]], acc[pilot[N0:]]
        raw, n = design_size(pooled_sd(a0, b0), margin, rule, N0)
        raw_vals[i], n_star[i] = raw, n
        rest = np.setdiff1d(np.arange(r), pilot, assume_unique=True)
        if 2 * n > len(rest):
            infeasible += 1
            continue
        conf = rng.choice(rest, size=2 * n, replace=False)
        eq, _ = tost_decision(acc[conf[:n]], acc[conf[n:]], margin, ALPHA)
        hits += int(eq)
    rate = hits / reps
    return {
        "achieved_rate": rate, "mc_se": mc_se(rate, reps),
        "infeasible_replications": int(infeasible),
        "n_star": _summary(n_star), "n_star_raw_median": float(np.median(raw_vals)),
        "share_floored": float(np.mean(raw_vals < FLOOR)),
        "total_runs_mean": float(np.mean(2 * N0 + 2 * n_star)),
        "total_runs_median": float(np.median(2 * N0 + 2 * n_star)),
    }


def two_stage_gaussian(sigma: float, margin: float, rule: str, reps: int,
                       rng: np.random.Generator) -> dict:
    hits = 0
    n_star = np.empty(reps, dtype=np.int64)
    for i in range(reps):
        a0, b0 = rng.normal(0.0, sigma, N0), rng.normal(0.0, sigma, N0)
        _, n = design_size(pooled_sd(a0, b0), margin, rule, N0)
        n_star[i] = n
        eq, _ = tost_decision(rng.normal(0.0, sigma, n), rng.normal(0.0, sigma, n), margin, ALPHA)
        hits += int(eq)
    rate = hits / reps
    return {"achieved_rate": rate, "mc_se": mc_se(rate, reps), "n_star": _summary(n_star),
            "total_runs_mean": float(np.mean(2 * N0 + 2 * n_star))}


def oracle_finite(acc: np.ndarray, margin: float, n: int, reps: int,
                  rng: np.random.Generator) -> dict:
    r = len(acc)
    hits = 0
    for _ in range(reps):
        idx = rng.choice(r, size=2 * n, replace=False)
        hits += int(tost_decision(acc[idx[:n]], acc[idx[n:]], margin, ALPHA)[0])
    rate = hits / reps
    return {"n": int(n), "achieved_rate": rate, "mc_se": mc_se(rate, reps)}


def reference_checks(acc: np.ndarray, s_full: float, margins, reps: int,
                     rng: np.random.Generator) -> dict:
    """The three comparators quoted next to the two-stage results.

    (i) Share of Gaussian pilots (5 + 5 draws with the full-pool s) whose point-estimate
    prescription falls below the full-pool n*, per margin: the chi-square mechanism alone.
    (ii) Probability that ten distinct pilot runs contain none of the lowest 5 % of the pool
    (exact hypergeometric, and the empirical share over the same pilots as (iii)).
    (iii) Share of pilots for which the upper 80 % bound of the pooled pilot variance covers the
    full-pool variance, on the real seed law and under a Gaussian law with the same s.
    """
    r = len(acc)
    factor = float(2 * (N0 - 1) / chi2.ppf(1.0 - UPPER_LEVEL, df=2 * (N0 - 1)))
    s2_full = s_full ** 2
    tail_cut = np.sort(acc)[int(np.ceil(0.05 * r)) - 1]
    gauss_below = {str(m): 0 for m in margins}
    n_full = {str(m): int(seeds_needed(m, POWER, s_full, 0.0, ALPHA)) for m in margins}
    cover_real = cover_gauss = tail_missed = 0
    for _ in range(reps):
        a, b = rng.normal(0.0, s_full, N0), rng.normal(0.0, s_full, N0)
        sp = pooled_sd(a, b)
        for m in margins:
            gauss_below[str(m)] += int(design_size(sp, m, "plugin", N0)[1] < n_full[str(m)])
        cover_gauss += int(factor * sp ** 2 >= s2_full)
        pilot = rng.choice(r, size=2 * N0, replace=False)
        cover_real += int(factor * pooled_sd(acc[pilot[:N0]], acc[pilot[N0:]]) ** 2 >= s2_full)
        tail_missed += int((acc[pilot] > tail_cut).all())
    k_tail = int(np.ceil(0.05 * r))
    return {
        "reps": int(reps),
        "gaussian_undersizing_share": {m: v / reps for m, v in gauss_below.items()},
        "tail_miss_probability_exact": comb(r - k_tail, 2 * N0) / comb(r, 2 * N0),
        "tail_miss_share_empirical": tail_missed / reps,
        "upper_bound_factor": factor,
        "upper_bound_covers_full_pool_variance": {"real_seed_law": cover_real / reps,
                                                  "gaussian": cover_gauss / reps},
    }


def _summary(values: np.ndarray) -> dict:
    return {"median": float(np.median(values)), "p2.5": float(np.percentile(values, 2.5)),
            "p97.5": float(np.percentile(values, 97.5)), "min": int(values.min()),
            "max": int(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--gauss-reps", type=int, default=20000)
    args = parser.parse_args()

    acc = load_seed_zoo()
    s_full = float(acc.std(ddof=1))
    rng = np.random.default_rng([ROOT_SEED, 7])
    out = {
        "provenance": {"date": _dt.date.today().isoformat(), "root_seed": ROOT_SEED,
                       "source": RUNS_CSV.name, "n_runs": int(len(acc)), "s_full": s_full,
                       "n0": N0, "power": POWER, "alpha": ALPHA, "floor": FLOOR,
                       "upper_bound_level": UPPER_LEVEL, "reps": args.reps,
                       "gauss_reps": args.gauss_reps},
        "by_margin": {},
    }
    for margin in MARGINS:
        n_full = int(seeds_needed(margin, POWER, s_full, 0.0, ALPHA))
        row = {"n_star_full": n_full,
               "oracle": oracle_finite(acc, margin, n_full, args.reps, rng),
               "rules": {}, "gaussian": {}}
        for rule in RULES:
            res = two_stage_finite(acc, margin, rule, args.reps, rng)
            res["share_below_full_n_star"] = None
            row["rules"][rule] = res
            row["gaussian"][rule] = two_stage_gaussian(s_full, margin, rule, args.gauss_reps, rng)
        out["by_margin"][str(margin)] = row

    # share of pilots that undersize relative to the full-pool answer (plug-in rule)
    rng2 = np.random.default_rng([ROOT_SEED, 8])
    for margin in MARGINS:
        n_full = out["by_margin"][str(margin)]["n_star_full"]
        below = 0
        for _ in range(args.reps):
            pilot = rng2.choice(len(acc), size=2 * N0, replace=False)
            _, n = design_size(pooled_sd(acc[pilot[:N0]], acc[pilot[N0:]]), margin, "plugin", N0)
            below += int(n < n_full)
        out["by_margin"][str(margin)]["rules"]["plugin"]["share_below_full_n_star"] = below / args.reps

    out["reference_checks"] = reference_checks(acc, s_full, MARGINS, args.gauss_reps,
                                               np.random.default_rng([ROOT_SEED, 9]))

    TABLES.mkdir(exist_ok=True)
    (TABLES / "case2_twostage.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# Two-stage design carried through on the seed zoo\n",
          f"Seed zoo: {len(acc)} runs, s = {s_full:.5f}; pilot n0 = {N0} per method, "
          f"target power {POWER}, alpha {ALPHA}, floor {FLOOR}; {args.reps} replications "
          f"(Gaussian reference {args.gauss_reps}).\n",
          "| margin | n* full | oracle rate at n* full | rule | achieved rate | n* median [2.5, 97.5] "
          "| share below n* full | mean total runs | Gaussian rate | Gaussian mean runs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for margin, row in out["by_margin"].items():
        for rule in RULES:
            r = row["rules"][rule]
            g = row["gaussian"][rule]
            share = r["share_below_full_n_star"]
            md.append(
                f"| {margin} | {row['n_star_full']} | {row['oracle']['achieved_rate']:.3f} "
                f"± {row['oracle']['mc_se']:.3f} | {rule} | {r['achieved_rate']:.3f} ± {r['mc_se']:.3f} "
                f"| {r['n_star']['median']:.0f} [{r['n_star']['p2.5']:.0f}, {r['n_star']['p97.5']:.0f}] "
                f"| {'' if share is None else f'{share:.3f}'} | {r['total_runs_mean']:.1f} "
                f"| {g['achieved_rate']:.3f} ± {g['mc_se']:.3f} | {g['total_runs_mean']:.1f} |")
    rc = out["reference_checks"]
    md.append("")
    md.append(f"Reference checks ({rc['reps']} pilots): Gaussian point-estimate undersizing share "
              + ", ".join(f"{m}: {v:.3f}" for m, v in rc["gaussian_undersizing_share"].items())
              + f"; P(ten pilot runs miss the lowest 5 %) exact {rc['tail_miss_probability_exact']:.3f}, "
              f"empirical {rc['tail_miss_share_empirical']:.3f}; upper 80 % bound (factor {rc['upper_bound_factor']:.3f}) "
              f"covers the full-pool variance in {rc['upper_bound_covers_full_pool_variance']['real_seed_law']:.3f} "
              f"of pilots on the real seed law and {rc['upper_bound_covers_full_pool_variance']['gaussian']:.3f} under a Gaussian law.")
    (TABLES / "case2_twostage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
