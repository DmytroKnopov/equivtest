"""power_exact_check.py: the shifted-t design heuristic against the exact
pooled-model TOST power (Shieh 2016), against the Welch-TOST on normal run
means, and against the realised rates of the two-level simulation.

Three reference values are placed next to the heuristic at every cell of the
power grid:

  exact         pooled TOST power under normal run means with a common
                variance (equivtest.power.tost_power_exact, an integral);
  welch_normal  Welch-TOST power under the same normal model, by Monte Carlo
                (the analysis uses the Welch test, whose random degrees of
                freedom are at most 2(n - 1), so it realises less power than
                the pooled test even on normal data);
  empirical     the realised rate of the two-level AUROC simulation stored in
                sim_power.json.

The differences heuristic - exact, exact - welch_normal and
empirical - welch_normal therefore separate the error of the shifted-t
approximation, the pooled-to-Welch gap and the departure of the two-level
AUROC process from normal run means. The exact run counts are also placed
next to the heuristic ones for every design table of the article.

Inputs (results/): sim_power.json, case_study.json,
case2_twostage.json, case2_paired.json, case2_modelzoo.json,
case2_multiberts.json. Outputs: results/power_exact_check.json and .md.

Usage (from repo root):
    python experiments_article2/power_exact_check.py [--reps N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equivtest.power import seeds_needed, tost_power, tost_power_exact

TABLES = Path(__file__).resolve().parents[1] / "results"
ALPHA = 0.05
DESIGN_MARGINS = [0.005, 0.0075, 0.01, 0.015, 0.02]
DESIGN_POWERS = [0.8, 0.9]
ROOT_SEED = 20261007
WELCH_REPS = 200_000
WELCH_REPS_CASE = 1_000_000


def welch_normal_power(
    delta_true: float, margin: float, n: int, tau: float, reps: int, seed, alpha: float = ALPHA
) -> tuple[float, float]:
    """Welch-TOST power on normal run means with a common variance tau^2, by
    Monte Carlo: returns the rate and its Monte-Carlo standard error."""
    rng = np.random.default_rng(seed)
    a = rng.normal(delta_true, tau, size=(reps, n))
    b = rng.normal(0.0, tau, size=(reps, n))
    diff = a.mean(axis=1) - b.mean(axis=1)
    va = a.var(axis=1, ddof=1) / n
    vb = b.var(axis=1, ddof=1) / n
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (n - 1) + vb**2 / (n - 1))
    t_crit = student_t.ppf(1.0 - alpha, df)
    rate = float(np.mean(np.abs(diff) < margin - t_crit * se))
    return rate, (rate * (1.0 - rate) / reps) ** 0.5


def _two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> float:
    """Score statistic for the equality of two binomial rates with the pooled
    estimate in the denominator. Unlike the Wald statistic it stays finite
    when one observed rate is 0 or 1; it is set to zero only when both rates
    sit at the same boundary, where the pooled variance is zero and the
    observed difference is zero as well."""
    x1, x2 = round(p1 * n1), round(p2 * n2)
    pooled = (x1 + x2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0
    return (p1 - p2) / (pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)) ** 0.5


def _summarise(subset: list[dict]) -> dict:
    if not subset:
        return {}
    dh = [r["heuristic"] - r["exact"] for r in subset]
    de = [r["empirical"] - r["exact"] for r in subset]
    out = {
        "cells": len(subset),
        "max_abs_heuristic_minus_exact": max(abs(x) for x in dh),
        "max_heuristic_minus_exact": max(dh),
        "min_heuristic_minus_exact": min(dh),
        "max_abs_empirical_minus_exact": max(abs(x) for x in de),
        "min_empirical_minus_exact": min(de),
        "max_empirical_minus_exact": max(de),
        "max_exact": max(r["exact"] for r in subset),
        "max_empirical": max(r["empirical"] for r in subset),
    }
    with_w = [r for r in subset if r.get("welch_normal") is not None]
    if with_w:
        ew = [r["exact"] - r["welch_normal"] for r in with_w]
        hw = [r["heuristic"] - r["welch_normal"] for r in with_w]
        mw = [r["empirical"] - r["welch_normal"] for r in with_w]
        abs_mw = [abs(x) for x in mw]
        # agreement of the realised rate with the Welch-normal rate as a
        # two-proportion score test (pooled estimate in the denominator), so
        # that cells with an observed rate of exactly one keep a finite
        # statistic; no cell is excluded
        zs = [
            _two_proportion_z(r["empirical"], r["emp_reps"], r["welch_normal"], r["welch_reps"])
            for r in with_w
        ]
        above = [r for r, z in zip(with_w, zs) if abs(z) > 3.0]
        out.update(
            {
                "max_exact_minus_welch": max(ew),
                "min_exact_minus_welch": min(ew),
                "max_heuristic_minus_welch": max(hw),
                "min_heuristic_minus_welch": min(hw),
                "max_abs_empirical_minus_welch": max(abs_mw),
                "median_abs_empirical_minus_welch": float(np.median(abs_mw)),
                "min_empirical_minus_welch": min(mw),
                "max_empirical_minus_welch": max(mw),
                "max_abs_z_empirical_vs_welch": max(abs(z) for z in zs),
                "cells_abs_z_above_3": len(above),
                "cells_abs_z_above_3_detail": [
                    {k: r[k] for k in ("sigma_seed", "n", "theta", "delta", "empirical", "welch_normal")}
                    for r in above
                ],
            }
        )
    return out


def grid_comparison(reps: int) -> dict:
    data = json.loads((TABLES / "sim_power.json").read_text(encoding="utf-8"))
    cells = data["cells"]
    margin = data["config"]["margin"]
    seeds = np.random.SeedSequence(ROOT_SEED).spawn(len(cells))
    rows = []
    for c, seed in zip(cells, seeds):
        exact = tost_power_exact(c["delta"], c["margin"], c["n"], c["sigma_seed"], c["v_bar"], c["alpha"])
        row = {
            "sigma_seed": c["sigma_seed"],
            "n": c["n"],
            "theta": c["theta"],
            "delta": c["delta"],
            "empirical": c["power_empirical"],
            "mc_se": c["power_mc_se"],
            "emp_reps": c["n_rep"],
            "heuristic": c["power_analytic_t"],
            "exact": exact,
            "welch_normal": None,
            "welch_mc_se": None,
            "welch_reps": reps,
        }
        if abs(c["delta"]) < margin:
            tau = (c["sigma_seed"] ** 2 + c["v_bar"]) ** 0.5
            row["welch_normal"], row["welch_mc_se"] = welch_normal_power(
                c["delta"], c["margin"], c["n"], tau, reps, seed, c["alpha"]
            )
        rows.append(row)

    def inside(r: dict) -> bool:
        return abs(r["delta"]) < margin

    def boundary(r: dict) -> bool:
        return abs(abs(r["delta"]) - margin) < 1e-12

    def outside(r: dict) -> bool:
        return abs(r["delta"]) > margin + 1e-12

    def remark_range(r: dict) -> bool:
        return inside(r) and r["n"] >= 5 and r["heuristic"] >= 0.2

    by_n = {}
    for n in sorted({r["n"] for r in rows}):
        sub = [r for r in rows if r["n"] == n]
        by_n[str(n)] = {
            "inside_margin": _summarise([r for r in sub if inside(r)]),
            "remark_range": _summarise([r for r in sub if remark_range(r)]),
            "at_boundary": _summarise([r for r in sub if boundary(r)]),
            "outside_margin": _summarise([r for r in sub if outside(r)]),
        }
    inside_ge5 = [r for r in rows if r["n"] >= 5 and inside(r)]
    above = [r for r in rows if r["heuristic"] - r["exact"] > 1e-10]
    return {
        "margin": margin,
        "alpha": data["config"]["alpha"],
        "welch_reps_per_cell": reps,
        "root_seed": ROOT_SEED,
        "by_n": by_n,
        "inside_margin_n_ge_5": _summarise(inside_ge5),
        "inside_margin_n_lt_5": _summarise([r for r in rows if r["n"] < 5 and inside(r)]),
        "remark_range_n_ge_5": _summarise([r for r in rows if remark_range(r)]),
        "heuristic_above_exact": {
            "cells": len(above),
            "deltas": sorted({r["delta"] for r in above}),
            "max_excess": max((r["heuristic"] - r["exact"] for r in above), default=0.0),
            "max_heuristic": max((r["heuristic"] for r in above), default=0.0),
            "max_exact": max((r["exact"] for r in above), default=0.0),
            "cells_inside_or_at_boundary": sum(1 for r in above if not outside(r)),
        },
        "worst_heuristic_cell_n_ge_5": max(inside_ge5, key=lambda r: abs(r["heuristic"] - r["exact"])),
        "worst_empirical_cell_n_ge_5": max(inside_ge5, key=lambda r: abs(r["empirical"] - r["exact"])),
        "cells": rows,
    }


def _design_cells(s: float, table: dict) -> dict:
    out = {}
    for m, by_power in table.items():
        out[m] = {}
        for p, published in by_power.items():
            out[m][p] = {
                "published": published,
                "heuristic": seeds_needed(float(m), float(p), s, 0.0, ALPHA),
                "exact": seeds_needed(float(m), float(p), s, 0.0, ALPHA, exact=True),
            }
    return out


def design_tables(reps_case: int) -> dict:
    case = json.loads((TABLES / "case_study.json").read_text(encoding="utf-8"))
    s_pooled = case["seed_level"]["s_pooled"]
    published = case["design_table"]
    table1 = {}
    for m in DESIGN_MARGINS:
        entry = {}
        for p in DESIGN_POWERS:
            n_h = seeds_needed(m, p, s_pooled, 0.0, ALPHA)
            n_e = seeds_needed(m, p, s_pooled, 0.0, ALPHA, exact=True)
            entry[str(p)] = {
                "heuristic": n_h,
                "exact": n_e,
                "published": published[str(m)][str(p)],
                "heuristic_power_at_heuristic_n": tost_power(0.0, m, n_h, s_pooled, 0.0, ALPHA),
                "exact_power_at_heuristic_n": tost_power_exact(0.0, m, n_h, s_pooled, 0.0, ALPHA),
            }
        table1[str(m)] = entry
    delta_hat = case["tost"]["delta"]
    seeds = np.random.SeedSequence(ROOT_SEED + 1).spawn(2)
    w2, w2_se = welch_normal_power(0.0, 0.01, 2, s_pooled, reps_case, seeds[0])
    w12, w12_se = welch_normal_power(0.0, 0.01, 12, s_pooled, reps_case, seeds[1])
    n2 = {
        "heuristic_delta0": tost_power(0.0, 0.01, 2, s_pooled, 0.0, ALPHA),
        "exact_delta0": tost_power_exact(0.0, 0.01, 2, s_pooled, 0.0, ALPHA),
        "exact_delta_hat": tost_power_exact(delta_hat, 0.01, 2, s_pooled, 0.0, ALPHA),
        "welch_normal_delta0": w2,
        "welch_normal_mc_se": w2_se,
        "empirical_delta0": case["empirical_power"]["n2"]["power"],
        "empirical_mc_se": case["empirical_power"]["n2"]["mc_se"],
    }
    n12 = {
        "heuristic": tost_power(0.0, 0.01, 12, s_pooled, 0.0, ALPHA),
        "exact": tost_power_exact(0.0, 0.01, 12, s_pooled, 0.0, ALPHA),
        "welch_normal": w12,
        "welch_normal_mc_se": w12_se,
        "empirical": case["empirical_power"]["n12"]["power"],
        "empirical_mc_se": case["empirical_power"]["n12"]["mc_se"],
        "heuristic_at_delta_hat": tost_power(delta_hat, 0.01, 12, s_pooled, 0.0, ALPHA),
        "exact_at_delta_hat": tost_power_exact(delta_hat, 0.01, 12, s_pooled, 0.0, ALPHA),
    }
    n_at_delta_hat = {
        "heuristic": seeds_needed(0.01, 0.8, s_pooled, 0.0, ALPHA, delta_true=delta_hat),
        "exact": seeds_needed(0.01, 0.8, s_pooled, 0.0, ALPHA, delta_true=delta_hat, exact=True),
    }

    two = json.loads((TABLES / "case2_twostage.json").read_text(encoding="utf-8"))
    s_full = two["provenance"]["s_full"]
    zoo = {}
    for m_key, rec in two["by_margin"].items():
        m = float(m_key)
        zoo[m_key] = {
            "published_n_star_full": rec["n_star_full"],
            "heuristic": seeds_needed(m, 0.8, s_full, 0.0, ALPHA),
            "exact": seeds_needed(m, 0.8, s_full, 0.0, ALPHA, exact=True),
            "exact_power_at_heuristic_n": tost_power_exact(0.0, m, rec["n_star_full"], s_full, 0.0, ALPHA),
            "oracle_achieved_rate": rec["oracle"]["achieved_rate"],
        }

    modelzoo = json.loads((TABLES / "case2_modelzoo.json").read_text(encoding="utf-8"))
    multiberts = json.loads((TABLES / "case2_multiberts.json").read_text(encoding="utf-8"))
    single = {
        "CIFAR-10 zoo": {
            "s": modelzoo["analysis_B"]["s"],
            "design_table": _design_cells(modelzoo["analysis_B"]["s"], modelzoo["analysis_A"]["design_table"]),
        }
    }
    for task, rec in multiberts["tasks"].items():
        sc = rec["single_configuration"]
        single["MultiBERTs, " + task] = {
            "s": sc["summary"]["sd"],
            "design_table": _design_cells(sc["summary"]["sd"], sc["design_table"]),
        }

    paired = json.loads((TABLES / "case2_paired.json").read_text(encoding="utf-8"))
    table4 = {}
    for name, rec in paired["comparisons"].items():
        plan = rec["n_star_independent_arms"]
        s = plan["s_pooled"]
        out = {"s_pooled": s}
        for key, val in plan.items():
            if key == "s_pooled":
                continue
            m = float(key)
            out[key] = {
                "published": val,
                "heuristic": seeds_needed(m, 0.8, s, 0.0, ALPHA),
                "exact": seeds_needed(m, 0.8, s, 0.0, ALPHA, exact=True),
            }
        dh = rec["delta_hat"]
        if abs(dh) < 0.01:
            try:
                out["at_delta_hat_0.01"] = {
                    "heuristic": seeds_needed(0.01, 0.8, s, 0.0, ALPHA, delta_true=dh),
                    "exact": seeds_needed(0.01, 0.8, s, 0.0, ALPHA, delta_true=dh, exact=True),
                }
            except ValueError:
                out["at_delta_hat_0.01"] = "unreachable"
        table4[name] = out

    return {
        "case_study_1": {
            "s_pooled": s_pooled,
            "delta_hat": delta_hat,
            "table1": table1,
            "n2": n2,
            "n12": n12,
            "n_star_80_at_delta_hat": n_at_delta_hat,
        },
        "zoo_two_stage": {"s_full": s_full, "by_margin": zoo},
        "single_configuration_design_tables": single,
        "table4_planning": table4,
    }


def write_markdown(result: dict, path: Path) -> None:
    g = result["grid"]
    lines = ["# Shifted-t heuristic against the exact pooled-model TOST power, the Welch-TOST on normal data and the realised rates", ""]
    lines.append(
        f"Grid of sim_power.json, margin {g['margin']}, alpha {g['alpha']}; 'inside' = cells with |Delta| < margin; "
        f"'remark range' = inside, n >= 5, heuristic >= 0.2; Welch-normal by Monte Carlo with {g['welch_reps_per_cell']} replications per cell (root seed {g['root_seed']})."
    )
    lines.append("")
    lines.append("## Inside the margin, by n")
    lines.append("")
    lines.append("| n | cells | max abs (heur - exact) | max (heur - exact) | max (exact - welch) | heur - welch min / max | emp - welch min / max | median abs (emp - welch) | max abs (emp - exact) | max abs z (emp vs welch) | cells abs z > 3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, rec in g["by_n"].items():
        s = rec["inside_margin"]
        lines.append(
            f"| {n} | {s['cells']} | {s['max_abs_heuristic_minus_exact']:.3f} | {s['max_heuristic_minus_exact']:+.4f} | {s['max_exact_minus_welch']:.3f} | "
            f"{s['min_heuristic_minus_welch']:+.3f} / {s['max_heuristic_minus_welch']:+.3f} | {s['min_empirical_minus_welch']:+.3f} / {s['max_empirical_minus_welch']:+.3f} | "
            f"{s['median_abs_empirical_minus_welch']:.4f} | {s['max_abs_empirical_minus_exact']:.3f} | {s['max_abs_z_empirical_vs_welch']:.2f} | {s['cells_abs_z_above_3']} |"
        )
    lines.append("")
    lines.append("## Remark range (inside, n >= 5, heuristic >= 0.2), by n")
    lines.append("")
    lines.append("| n | cells | max (exact - heur) | max (exact - welch) | heur - welch min / max | emp - welch min / max | max abs (emp - exact) |")
    lines.append("|---|---|---|---|---|---|---|")
    for n, rec in g["by_n"].items():
        s = rec["remark_range"]
        if not s:
            continue
        lines.append(
            f"| {n} | {s['cells']} | {-s['min_heuristic_minus_exact']:.3f} | {s['max_exact_minus_welch']:.3f} | "
            f"{s['min_heuristic_minus_welch']:+.3f} / {s['max_heuristic_minus_welch']:+.3f} | {s['min_empirical_minus_welch']:+.3f} / {s['max_empirical_minus_welch']:+.3f} | {s['max_abs_empirical_minus_exact']:.3f} |"
        )
    r = g["remark_range_n_ge_5"]
    lines.append("")
    lines.append(
        f"Remark range overall: max (exact - heur) {-r['min_heuristic_minus_exact']:.4f}; max (exact - welch) {r['max_exact_minus_welch']:.4f}; "
        f"heur - welch from {r['min_heuristic_minus_welch']:+.4f} to {r['max_heuristic_minus_welch']:+.4f}; emp - welch from {r['min_empirical_minus_welch']:+.4f} to {r['max_empirical_minus_welch']:+.4f} "
        f"(max abs {r['max_abs_empirical_minus_welch']:.4f}, median abs {r['median_abs_empirical_minus_welch']:.4f}, max abs z {r['max_abs_z_empirical_vs_welch']:.2f}, cells abs z > 3: {r['cells_abs_z_above_3']}); "
        f"emp - exact from {r['min_empirical_minus_exact']:+.4f} to {r['max_empirical_minus_exact']:+.4f}."
    )
    a = g["inside_margin_n_ge_5"]
    b = g["inside_margin_n_lt_5"]
    lines.append(
        f"Inside, n >= 5 (all power levels): max |heur - exact| {a['max_abs_heuristic_minus_exact']:.4f}, heur - welch from {a['min_heuristic_minus_welch']:+.4f} to {a['max_heuristic_minus_welch']:+.4f}, "
        f"max |emp - exact| {a['max_abs_empirical_minus_exact']:.4f}, max abs z (emp vs welch) {a['max_abs_z_empirical_vs_welch']:.2f}, cells abs z > 3: {a['cells_abs_z_above_3']}."
    )
    lines.append(
        f"Inside, n < 5: max |heur - exact| {b['max_abs_heuristic_minus_exact']:.4f}, max |emp - exact| {b['max_abs_empirical_minus_exact']:.4f}, max (exact - welch) {b['max_exact_minus_welch']:.4f}, "
        f"max abs z (emp vs welch) {b['max_abs_z_empirical_vs_welch']:.2f}, cells abs z > 3: {b['cells_abs_z_above_3']}."
    )
    allin = [r for r in g["cells"] if r.get("welch_normal") is not None]
    zs = [abs(_two_proportion_z(r["empirical"], r["emp_reps"], r["welch_normal"], r["welch_reps"])) for r in allin]
    lines.append(f"All {len(allin)} inside cells: max abs z (emp vs welch) {max(zs):.2f}, cells abs z > 3: {sum(1 for z in zs if z > 3.0)}.")
    h = g["heuristic_above_exact"]
    lines.append(
        f"Cells where the heuristic exceeds the exact power: {h['cells']} (Delta values {h['deltas']}; max excess {h['max_excess']:.5f}; "
        f"largest heuristic {h['max_heuristic']:.5f}, largest exact {h['max_exact']:.6f}; inside or at the boundary: {h['cells_inside_or_at_boundary']})."
    )
    w = g["worst_heuristic_cell_n_ge_5"]
    lines.append(f"Worst heuristic cell (n >= 5, inside): sigma_seed {w['sigma_seed']}, n {w['n']}, theta {w['theta']}, Delta {w['delta']}: heur {w['heuristic']:.3f}, exact {w['exact']:.3f}, welch {w['welch_normal']:.3f}, emp {w['empirical']:.3f}")
    w = g["worst_empirical_cell_n_ge_5"]
    lines.append(f"Worst empirical cell (n >= 5, inside): sigma_seed {w['sigma_seed']}, n {w['n']}, theta {w['theta']}, Delta {w['delta']}: heur {w['heuristic']:.3f}, exact {w['exact']:.3f}, welch {w['welch_normal']:.3f}, emp {w['empirical']:.3f}")
    lines.append("")
    lines.append("Boundary cells (|Delta| = margin): max exact / max empirical by n")
    for n, rec in g["by_n"].items():
        s = rec["at_boundary"]
        lines.append(f"- n = {n}: exact {s['max_exact']:.4f}, empirical {s['max_empirical']:.4f}")
    lines.append("")

    d = result["design"]
    c1 = d["case_study_1"]
    lines.append(f"## Table 2 (first case study, s = {c1['s_pooled']:.4f}, Delta = 0)")
    lines.append("")
    lines.append("| margin | n* 80% heur | n* 80% exact | n* 90% heur | n* 90% exact | exact power at heur n (80%) |")
    lines.append("|---|---|---|---|---|---|")
    for m, e in c1["table1"].items():
        lines.append(
            f"| {m} | {e['0.8']['heuristic']} | {e['0.8']['exact']} | {e['0.9']['heuristic']} | {e['0.9']['exact']} | {e['0.8']['exact_power_at_heuristic_n']:.3f} |"
        )
    lines.append("")
    n2 = c1["n2"]
    lines.append(
        f"n = 2: heuristic {n2['heuristic_delta0']:.3f}, exact {n2['exact_delta0']:.4f} (at Delta-hat {n2['exact_delta_hat']:.4f}), "
        f"welch-normal {n2['welch_normal_delta0']:.4f} +- {n2['welch_normal_mc_se']:.4f}, empirical {n2['empirical_delta0']:.3f} +- {n2['empirical_mc_se']:.3f}"
    )
    n12 = c1["n12"]
    lines.append(
        f"n = 12: heuristic {n12['heuristic']:.3f}, exact {n12['exact']:.3f}, welch-normal {n12['welch_normal']:.4f} +- {n12['welch_normal_mc_se']:.4f}, "
        f"empirical {n12['empirical']:.3f} +- {n12['empirical_mc_se']:.3f}; at Delta-hat heuristic {n12['heuristic_at_delta_hat']:.3f}, exact {n12['exact_at_delta_hat']:.3f}"
    )
    nd = c1["n_star_80_at_delta_hat"]
    lines.append(f"n* (80%) at Delta-hat: heuristic {nd['heuristic']}, exact {nd['exact']}")
    lines.append("")
    z = d["zoo_two_stage"]
    lines.append(f"## Zoo two-stage margins (s_full = {z['s_full']:.4f})")
    for m, e in z["by_margin"].items():
        lines.append(
            f"- margin {m}: published {e['published_n_star_full']}, heuristic {e['heuristic']}, exact {e['exact']}, "
            f"exact power at published n {e['exact_power_at_heuristic_n']:.3f}, oracle realised {e['oracle_achieved_rate']:.3f}"
        )
    lines.append("")
    lines.append("## Single-configuration design tables (Table 3 prints margin 0.01, power 0.8)")
    for name, e in d["single_configuration_design_tables"].items():
        parts = []
        for m, by_p in e["design_table"].items():
            for p, v in by_p.items():
                flag = "" if v["heuristic"] == v["exact"] else " (differs)"
                parts.append(f"{m}/{p}: {v['heuristic']} vs {v['exact']}{flag}")
        lines.append(f"- {name} (s {e['s']:.4f}): " + "; ".join(parts))
    lines.append("")
    lines.append("## Table 4 planning column (independent arms, Delta = 0 unless stated)")
    for name, e in d["table4_planning"].items():
        parts = [f"s {e['s_pooled']:.4f}"]
        for k, v in e.items():
            if k == "s_pooled":
                continue
            if isinstance(v, dict):
                pub = f", published {v['published']}" if "published" in v else ""
                parts.append(f"{k}: heur {v['heuristic']}, exact {v['exact']}{pub}")
            else:
                parts.append(f"{k}: {v}")
        lines.append(f"- {name}: " + "; ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=WELCH_REPS, help="Monte-Carlo replications per grid cell for the Welch-normal power")
    parser.add_argument("--reps-case", type=int, default=WELCH_REPS_CASE, help="replications for the two case-study cells")
    args = parser.parse_args()
    result = {"grid": grid_comparison(args.reps), "design": design_tables(args.reps_case)}
    (TABLES / "power_exact_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_markdown(result, TABLES / "power_exact_check.md")
    print((TABLES / "power_exact_check.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
