"""case2_paired.py -- paired analysis of the eight seed-matched comparisons.

Second case study of article 2, companion to case2_modelzoo.py (Analysis C)
and case2_multiberts.py (comparisons i and ii). All eight two-method
comparisons of the case study are matched by seed: the two hyperparameter-zoo
pairs share the seeds 1..10 of their arms, the MultiBERTs learning-rate and
pretraining-step comparisons share the pretraining seed. The observed data are
therefore analysed with the paired seed-level TOST, the one-sample t-based
TOST on the within-seed differences; the unpaired Welch-TOST of the two
scripts above is kept as a sensitivity check, and the independent-arm design
count n* (pooled s of the two arms, Delta = 0) is kept as the planning answer
for a fresh experiment with independent arms.

Accuracies are counts over the evaluation set (k / m). The differences are
formed on the integer counts and divided by m once, so that two equal
differences are bit-identical floats and a draw whose differences all coincide
is recognised exactly as degenerate (no standard error; counted as a
non-certification). The mean difference is also reported as an exact decimal
rounded half-to-even to four places, which the table uses.

Per comparison: paired Delta-hat, paired 90 % interval, paired TOST p-values
and verdicts at the margins 0.01 and 0.02, the same-seed correlation, the
unpaired Welch-TOST for comparison, the independent-arm n*, and the
share of the subsets of n matched pairs (n = 2, 3, 5) in which the paired TOST
at delta = 0.01 certifies, enumerated exactly over all C(10, n) or C(25, n)
subsets (Monte Carlo only if a pool ever exceeds EXACT_MAX subsets); not
computed for the five-pair comparisons.

Reads results/case2_modelzoo.json and case2_multiberts.json; writes
results/case2_paired.json and case2_paired.md.

Usage (from repo root):
    python experiments_article2/case2_paired.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case2_common import ALPHA, ROOT_SEED, mc_se, paired_tost, tost_dict  # noqa: E402
from equivtest.power import seeds_needed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results"

MARGINS = [0.01, 0.02]
SUB_MARGIN = 0.01
N_GRID = [2, 3, 5]
POWER = 0.8
EXACT_MAX = 200_000  # enumerate all subsets up to this many; Monte Carlo beyond


def counts(acc, m: int) -> np.ndarray:
    """Integer correct counts behind accuracies k / m; fails if any is not one."""
    scaled = np.asarray(acc, dtype=np.float64) * m
    k = np.rint(scaled)
    if np.abs(scaled - k).max() > 1e-6:
        raise ValueError("an accuracy is not an integer count over m")
    return k.astype(np.int64)


def exact_mean_4dp(diff_counts: np.ndarray, m: int) -> str:
    frac = Fraction(int(diff_counts.sum()), m * len(diff_counts))
    dec = Decimal(frac.numerator) / Decimal(frac.denominator)
    return f"{dec.quantize(Decimal('0.0001'), rounding=ROUND_HALF_EVEN):+}"


def paired_subsampling(d_all: np.ndarray, n_grid, margin: float, reps: int,
                       rng: np.random.Generator) -> dict:
    """Share of subsets of n matched pairs in which the paired TOST certifies.

    The pools are small (10 or 25 pairs), so the subsets are enumerated exactly
    whenever their number is at most EXACT_MAX (all cases of the case study:
    C(10, 5) = 252, C(25, 5) = 53 130); Monte Carlo without replacement is the
    fallback for larger pools. A subset whose differences all coincide has no
    standard error and counts as a non-certification.
    """
    k = len(d_all)
    out = {"reps": int(reps), "pairs_available": int(k), "by_n": {}}
    for n in n_grid:
        if k <= 5 or n >= k:
            out["by_n"][str(n)] = {"skipped": f"n = {n} is not computed for {k} pairs or fewer"}
            continue
        total = comb(k, n)
        hits = 0
        degenerate = 0
        if total <= EXACT_MAX:
            for idx in combinations(range(k), n):
                res = paired_tost(d_all[list(idx)], margin, ALPHA)
                degenerate += int(res.get("degenerate", False))
                hits += int(res["equivalent"])
            rate = hits / total
            out["by_n"][str(n)] = {"certification_rate": rate, "exact": True, "n_subsets": int(total),
                                   "degenerate_subsets": int(degenerate),
                                   "degenerate_share": degenerate / total}
        else:
            for _ in range(reps):
                d = d_all[rng.choice(k, size=n, replace=False)]
                res = paired_tost(d, margin, ALPHA)
                degenerate += int(res.get("degenerate", False))
                hits += int(res["equivalent"])
            rate = hits / reps
            out["by_n"][str(n)] = {"certification_rate": rate, "exact": False, "mc_se": mc_se(rate, reps),
                                   "degenerate_draws": int(degenerate),
                                   "degenerate_share": degenerate / reps}
    return out


def analyse(name: str, a, b, seeds, m: int, reps: int, rng: np.random.Generator) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"{name}: arms of unequal length cannot be paired")
    diff_counts = counts(a, m) - counts(b, m)
    d = diff_counts / m
    s_pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    return {
        "n_pairs": int(len(d)),
        "m_items": int(m),
        "seeds": list(seeds) if seeds is not None else None,
        "delta_hat": float(d.mean()),
        "delta_hat_4dp": exact_mean_4dp(diff_counts, m),
        "sd_differences": float(d.std(ddof=1)),
        "same_seed_correlation": float(np.corrcoef(a, b)[0, 1]),
        "paired": {str(mg): paired_tost(d, mg, ALPHA) for mg in MARGINS},
        "unpaired": {str(mg): tost_dict(a, b, mg) for mg in MARGINS},
        "n_star_independent_arms": {
            "s_pooled": s_pooled,
            str(SUB_MARGIN): int(seeds_needed(SUB_MARGIN, POWER, s_pooled, 0.0, ALPHA)),
        },
        "paired_subsampling": paired_subsampling(d, N_GRID, SUB_MARGIN, reps, rng),
    }


def main() -> None:
    zoo_all = json.load(open(TABLES / "case2_modelzoo.json", encoding="utf-8"))
    zoo, m_zoo = zoo_all["analysis_C"], int(zoo_all["provenance"]["m_items"])
    mb = json.load(open(TABLES / "case2_multiberts.json", encoding="utf-8"))["tasks"]
    rng = np.random.default_rng([ROOT_SEED, 11])
    out = {"provenance": {"date": _dt.date.today().isoformat(), "root_seed": ROOT_SEED,
                          "alpha": ALPHA, "margins": MARGINS, "subsampling_margin": SUB_MARGIN,
                          "n_grid": N_GRID, "reps_zoo": 5000, "reps_multiberts": 2000},
           "comparisons": {}}
    for key, label in (("P1_adam_vs_sgd", "zoo Adam vs SGD"), ("P2_gelu_vs_tanh", "zoo GELU vs tanh")):
        p = zoo[key]
        if p["seeds_a"] != p["seeds_b"]:
            raise ValueError(f"{key}: seed labels of the two arms differ")
        out["comparisons"][label] = analyse(label, p["acc_a"], p["acc_b"], p["seeds_a"], m_zoo, 5000, rng)
    for task in ("RTE", "SST-2", "MNLI"):
        c = mb[task]["comparison_i_lr"]
        out["comparisons"][f"{task} lr"] = analyse(f"{task} lr", c["acc_a"], c["acc_b"], None,
                                                   int(mb[task]["m"]), 2000, rng)
    for task in ("RTE", "SST-2", "MNLI"):
        c = mb[task]["comparison_ii_steps"]
        out["comparisons"][f"{task} steps"] = analyse(f"{task} steps", c["acc_a"], c["acc_b"],
                                                      c.get("pretrain_ids"), int(mb[task]["m"]), 2000, rng)

    TABLES.mkdir(exist_ok=True)
    (TABLES / "case2_paired.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# Paired analysis of the eight seed-matched comparisons\n",
          "| comparison | pairs | delta-hat (exact, 4 dp) | paired 90 % CI | paired p 0.01 | paired p 0.02 "
          "| verdict 0.01 / 0.02 | corr | unpaired p 0.01 / 0.02 | unpaired 90 % CI | n* (indep. arms, 0.01) "
          "| paired rate n=2 (degenerate) | n=3 (degenerate) | n=5 (degenerate) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in out["comparisons"].items():
        p1, p2 = r["paired"]["0.01"], r["paired"]["0.02"]
        u1, u2 = r["unpaired"]["0.01"], r["unpaired"]["0.02"]
        sub = r["paired_subsampling"]["by_n"]

        def rate(n):
            row = sub[str(n)]
            if "skipped" in row:
                return "-"
            if row.get("exact"):
                return f"{row['certification_rate']:.4f} (exact, {row['n_subsets']} subsets, {row['degenerate_subsets']} degenerate)"
            return f"{row['certification_rate']:.4f} (MC, {row['degenerate_draws']} degenerate)"

        def fmt_p(x):
            return "-" if x is None else f"{x:.3f}"

        md.append(
            f"| {label} | {r['n_pairs']} | {r['delta_hat_4dp']} | [{p1['ci90'][0]:+.4f}, {p1['ci90'][1]:+.4f}] "
            f"| {fmt_p(p1['p_tost'])} | {fmt_p(p2['p_tost'])} | {'yes' if p1['equivalent'] else 'no'} / "
            f"{'yes' if p2['equivalent'] else 'no'} | {r['same_seed_correlation']:+.3f} "
            f"| {u1['p_tost']:.3f} / {u2['p_tost']:.3f} | [{u1['ci90'][0]:+.4f}, {u1['ci90'][1]:+.4f}] "
            f"| {r['n_star_independent_arms']['0.01']} | {rate(2)} | {rate(3)} | {rate(5)} |")
    (TABLES / "case2_paired.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
