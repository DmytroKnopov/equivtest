"""case_study.py — the equivalence protocol applied to the case-study data.

Demonstration on the four training runs of the accompanying article (JEM on
CIFAR-10; SGLD vs Predictor-Corrector samplers, two seeds per method).
What it produces:

  1. Sanity: per-run AUROCs, macro-5 and the seed-level Welch-TOST recomputed
     from the raw per-image scores with `equivtest`, checked against the
     published table `data/hier_bootstrap_tost.json` (the hierarchical-
     bootstrap CI is replayed bit-exactly by tests/test_hierboot.py and is
     quoted from that table rather than recomputed here).
  2. Estimates: seed-level dispersion per method and pooled s^2;
     image-level (DeLong) variance of the macro-5 statistic per run including
     the cross-dataset covariance induced by the shared ID sample; between-run
     per-image score correlation per dataset.
  3. Power answers: retrospective TOST power of the n = 2 design (analytic
     and, unless --fast, empirical from a targeted two-level simulation
     calibrated to the estimates above), the required run count n* for
     delta = 0.01 at 80%/90% power, a small design table over margins, and a
     chi^2 interval for n* reflecting the ~1.2 effective df of pooled s^2.

Output: results/case_study.json + case_study.md.

Usage (from repo root):
    python experiments/case_study.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equivtest.auroc import auroc, delong_variance
from equivtest.dgp import simulate_experiment
from equivtest.power import seeds_needed, tost_power
from equivtest.tost import welch_tost

SCORES_DIR = Path("data/ood_scores")
PUBLISHED_JSON = Path("data/hier_bootstrap_tost.json")
OUT_DIR = Path("results")

DATASETS = ["svhn", "cifar100", "dtd", "lsun", "isun"]

# The four case-study runs.
SGLD_RUNS = [
    ("SGLD, seed 42", "sgld_seed42"),
    ("SGLD, seed 123", "sgld_seed123"),
]
PC_RUNS = [
    ("PC, seed 42", "pc_seed42"),
    ("PC, seed 123", "pc_seed123"),
]

MARGIN = 0.01
ALPHA = 0.05
POWER_TARGETS = (0.80, 0.90)
DESIGN_MARGINS = (0.005, 0.0075, 0.01, 0.015, 0.02)

# Targeted empirical power check (single binormal OOD set as proxy).
SIM_M0 = SIM_M1 = 10_000
SIM_REPS = 2000
SIM_SEED = 20260709


def _path(stem: str, ds: str) -> Path:
    return SCORES_DIR / f"{stem}__{ds}.npz"


def load_scores(path: Path) -> np.ndarray:
    with np.load(path) as f:
        return f["scores"].astype(np.float64)


def load_run(label: str, stem: str) -> dict:
    id_scores = load_scores(_path(stem, "cifar10"))
    out = {ds: load_scores(_path(stem, ds)) for ds in DATASETS}
    per_ds = {ds: auroc(id_scores, out[ds]) for ds in DATASETS}
    macro5 = float(np.mean([per_ds[ds] for ds in DATASETS]))
    return {"label": label, "stem": stem, "in": id_scores, "out": out,
            "auroc": per_ds, "macro5": macro5}


def id_placements(id_scores: np.ndarray, ood_scores: np.ndarray) -> np.ndarray:
    """Placement value P(ood > id_k) per ID image, ties as 1/2."""
    ood_sorted = np.sort(ood_scores)
    m1 = len(ood_sorted)
    lo = np.searchsorted(ood_sorted, id_scores, side="left")
    hi = np.searchsorted(ood_sorted, id_scores, side="right")
    return 1.0 - (lo + hi) / (2.0 * m1)


def macro5_image_variance(run: dict) -> tuple[float, dict]:
    """DeLong variance of the macro-5 statistic conditional on the run.

    macro5 = mean of 5 per-dataset AUROCs that share the ID sample, so
    Var(macro5) = (sum of OOD-side terms + 1'C1/m0) / 25, where C is the
    5x5 covariance matrix of the per-ID-image placement values (the OOD
    samples are disjoint across datasets and contribute no cross terms).
    """
    m0 = len(run["in"])
    ood_side = 0.0
    per_ds_v = {}
    placements = []
    for ds in DATASETS:
        v, s10, s01 = delong_variance(run["in"], run["out"][ds])
        per_ds_v[ds] = v
        ood_side += s10 / len(run["out"][ds])
        p = id_placements(run["in"], run["out"][ds])
        assert np.isclose(p.mean(), run["auroc"][ds], atol=1e-12)
        assert np.isclose(np.var(p, ddof=1), s01, atol=1e-12)
        placements.append(p)
    cov01 = np.cov(np.vstack(placements), ddof=1)
    v_macro = float((ood_side + cov01.sum() / m0) / 25.0)
    return v_macro, per_ds_v


def score_correlations(all_runs: list[dict]) -> dict:
    """Pearson correlation of per-image scores over the 6 run pairs."""
    out = {}
    for key in ["cifar10"] + DATASETS:
        vecs = [r["in"] if key == "cifar10" else r["out"][key] for r in all_runs]
        cors = [float(np.corrcoef(vecs[i], vecs[j])[0, 1])
                for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        out[key] = {"mean": float(np.mean(cors)),
                    "min": float(np.min(cors)), "max": float(np.max(cors))}
    return out


def hanley_mcneil_v(theta: float, m0: int, m1: int) -> float:
    q1 = theta / (2.0 - theta)
    q2 = 2.0 * theta**2 / (1.0 + theta)
    return (theta * (1.0 - theta) + (m1 - 1) * (q1 - theta**2)
            + (m0 - 1) * (q2 - theta**2)) / (m0 * m1)


def empirical_tost_power(n: int, sigma_sim: float, theta: float, rho: float,
                         reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Rejection rate of the seed-level TOST in the calibrated two-level model."""
    hits = 0
    t0 = time.perf_counter()
    for i in range(reps):
        runs_a, runs_b = simulate_experiment(
            theta, theta, sigma_sim, n, SIM_M0, SIM_M1, rng, image_rho=rho
        )
        th_a = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_a])
        th_b = np.array([auroc(r.id_scores, r.ood_scores) for r in runs_b])
        hits += int(welch_tost(th_a, th_b, MARGIN, ALPHA).equivalent)
        if (i + 1) % 500 == 0:
            print(f"    [{i + 1}/{reps}] {time.perf_counter() - t0:.0f}s", flush=True)
    p = hits / reps
    return p, float(np.sqrt(p * (1 - p) / reps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="skip the empirical power simulations")
    args = parser.parse_args()

    print("Loading the four runs...")
    sgld = [load_run(label, stem) for label, stem in SGLD_RUNS]
    pc = [load_run(label, stem) for label, stem in PC_RUNS]
    all_runs = sgld + pc

    m0 = len(sgld[0]["in"])
    m1 = {ds: len(sgld[0]["out"][ds]) for ds in DATASETS}
    print(f"  ID sample: CIFAR-10 test, m0 = {m0}")
    print(f"  OOD sizes: " + ", ".join(f"{ds}={m1[ds]}" for ds in DATASETS))

    print("\nPer-run AUROCs:")
    header = f"  {'run':<16}" + "".join(f"{ds:>10}" for ds in DATASETS) + f"{'macro-5':>10}"
    print(header)
    for r in all_runs:
        row = f"  {r['label']:<16}" + "".join(f"{r['auroc'][ds]:>10.4f}" for ds in DATASETS)
        print(row + f"{r['macro5']:>10.4f}")

    sgld_macro = np.array([r["macro5"] for r in sgld])
    pc_macro = np.array([r["macro5"] for r in pc])

    # --- seed-level dispersion -------------------------------------------------
    s2_sgld = float(np.var(sgld_macro, ddof=1))
    s2_pc = float(np.var(pc_macro, ddof=1))
    s2_pooled = (s2_sgld + s2_pc) / 2.0
    print(f"\nSeed-level macro-5 dispersion (1 df each):")
    print(f"  s_SGLD = {np.sqrt(s2_sgld):.4f}   s_PC = {np.sqrt(s2_pc):.4f}"
          f"   pooled s = {np.sqrt(s2_pooled):.4f}  (s^2 = {s2_pooled:.3e})")

    # --- image-level (DeLong) variance of macro-5 ------------------------------
    print("\nImage-level (DeLong) variance of macro-5 per run:")
    v_macro_runs = {}
    v_per_ds_runs = {}
    for r in all_runs:
        v_macro, per_ds_v = macro5_image_variance(r)
        v_macro_runs[r["label"]] = v_macro
        v_per_ds_runs[r["label"]] = per_ds_v
        print(f"  {r['label']:<16} v_macro = {v_macro:.3e}")
    v_bar_macro = float(np.mean(list(v_macro_runs.values())))
    print(f"  mean v_bar = {v_bar_macro:.3e}  ->  v_bar / s^2 = {v_bar_macro / s2_pooled:.1%}")

    # --- between-run score correlation -----------------------------------------
    rho = score_correlations(all_runs)
    print("\nBetween-run per-image score correlation (mean over 6 pairs):")
    for key, c in rho.items():
        print(f"  {key:<9} mean = {c['mean']:.3f}  range [{c['min']:.3f}, {c['max']:.3f}]")
    rho_mean = float(np.mean([c["mean"] for c in rho.values()]))

    # --- TOST reproduction ------------------------------------------------------
    tost = welch_tost(pc_macro, sgld_macro, MARGIN, ALPHA)  # delta = PC - SGLD
    print(f"\nSeed-level Welch-TOST (margin = {MARGIN}):")
    print(f"  delta = {tost.delta:+.4f}  SE = {tost.se:.4f}  df = {tost.df:.4f}")
    print(f"  p_tost = {tost.p_tost:.4f}  equivalent = {tost.equivalent}")
    print(f"  90% CI = [{tost.ci90[0]:+.4f}, {tost.ci90[1]:+.4f}]")

    published = None
    if PUBLISHED_JSON.exists():
        published = json.loads(PUBLISHED_JSON.read_text(encoding="utf-8"))
        pub_tost = published["tost"]
        assert np.allclose(sorted(sgld_macro),
                           sorted(published["point_estimates"]["sgld_macro5"]), atol=1e-9)
        assert np.allclose(sorted(pc_macro),
                           sorted(published["point_estimates"]["pc_macro5"]), atol=1e-9)
        assert np.isclose(tost.delta, pub_tost["diff"], atol=1e-12)
        assert np.isclose(tost.se, pub_tost["se"], atol=1e-12)
        assert np.isclose(tost.df, pub_tost["df"], atol=1e-9)
        assert np.isclose(tost.p_tost, pub_tost["p_tost"], atol=1e-9)
        hb = published["hierarchical_bootstrap"]["macro5"]
        print(f"  matches tables/hier_bootstrap_tost.json (published p = {pub_tost['p_tost']:.4f});"
              f" published hierarchical CI = [{hb['ci_low']:+.4f}, {hb['ci_high']:+.4f}]")
    else:
        print("  WARNING: data/hier_bootstrap_tost.json not found -- sanity check skipped")

    # --- power: retrospective and design ----------------------------------------
    # The pooled s^2 already estimates the whole effective per-run variance on
    # the shared test set, so it is passed as sigma_seed with v_bar = 0.
    s_pooled = float(np.sqrt(s2_pooled))
    retro = {
        "power_t_delta0": tost_power(0.0, MARGIN, 2, s_pooled, 0.0, ALPHA, use_t=True),
        "power_normal_delta0": tost_power(0.0, MARGIN, 2, s_pooled, 0.0, ALPHA, use_t=False),
        "power_t_deltahat": tost_power(tost.delta, MARGIN, 2, s_pooled, 0.0, ALPHA, use_t=True),
    }
    print(f"\nRetrospective TOST power of the case-study design (n = 2, delta_true = 0):")
    print(f"  shifted-t: {retro['power_t_delta0']:.4f}   normal: {retro['power_normal_delta0']:.4f}"
          f"   at observed delta ({tost.delta:+.4f}): {retro['power_t_deltahat']:.4f}")

    design = {}
    for margin in DESIGN_MARGINS:
        design[margin] = {pw: seeds_needed(margin, pw, s_pooled, 0.0, ALPHA)
                          for pw in POWER_TARGETS}
    n80 = design[MARGIN][0.80]
    n90 = design[MARGIN][0.90]
    print(f"\nRuns per method needed at delta = {MARGIN}: n* = {n80} (80%), {n90} (90%)")
    print("Design table n*(margin, power):")
    print(f"  {'margin':>8}" + "".join(f"{int(pw * 100):>7}%" for pw in POWER_TARGETS))
    for margin in DESIGN_MARGINS:
        print(f"  {margin:>8}" + "".join(f"{design[margin][pw]:>8}" for pw in POWER_TARGETS))

    # chi^2 interval for n*(delta=0.01, 80%) from the sampling noise of s^2:
    # SE^2 of the TOST equals s2_pooled at n=2, so the Welch-Satterthwaite df
    # of the pooled estimate is the TOST df (~1.21).
    df_s2 = tost.df
    s2_lo = df_s2 * s2_pooled / chi2.ppf(0.975, df_s2)
    s2_hi = df_s2 * s2_pooled / chi2.ppf(0.025, df_s2)
    n80_lo = seeds_needed(MARGIN, 0.80, float(np.sqrt(s2_lo)), 0.0, ALPHA)
    n80_hi = seeds_needed(MARGIN, 0.80, float(np.sqrt(s2_hi)), 0.0, ALPHA)
    print(f"\n95% chi^2 interval for s^2 (df = {df_s2:.2f}): [{s2_lo:.2e}, {s2_hi:.2e}]"
          f"  ->  n*(80%) in [{n80_lo}, {n80_hi}]")

    # --- empirical check of the two headline powers ------------------------------
    empirical = None
    if not args.fast:
        theta_mean = float(np.mean([r["macro5"] for r in all_runs]))
        v_sim = hanley_mcneil_v(theta_mean, SIM_M0, SIM_M1)
        # Match the between-seed variance of theta-hat in the simulation to the
        # observed pooled s^2: sigma_sim^2 + (1 - rho) v_sim = s2_pooled.
        sigma_sim = float(np.sqrt(max(s2_pooled - (1.0 - rho_mean) * v_sim, 0.0)))
        print(f"\nEmpirical power (binormal proxy: theta = {theta_mean:.3f}, "
              f"rho = {rho_mean:.2f}, sigma_sim = {sigma_sim:.4f}, "
              f"m0 = m1 = {SIM_M0}, {SIM_REPS} replications):")
        rng = np.random.default_rng(SIM_SEED)
        emp2, se2 = empirical_tost_power(2, sigma_sim, theta_mean, rho_mean, SIM_REPS, rng)
        print(f"  n = 2:   empirical = {emp2:.4f} (MC SE {se2:.4f})"
              f"   analytic (t) = {retro['power_t_delta0']:.4f}")
        empn, sen = empirical_tost_power(n80, sigma_sim, theta_mean, rho_mean, SIM_REPS, rng)
        power_at_n80 = tost_power(0.0, MARGIN, n80, s_pooled, 0.0, ALPHA)
        print(f"  n = {n80}:  empirical = {empn:.4f} (MC SE {sen:.4f})"
              f"   analytic (t) = {power_at_n80:.4f}")
        empirical = {
            "theta": theta_mean, "rho": rho_mean, "sigma_sim": sigma_sim,
            "m0": SIM_M0, "m1": SIM_M1, "reps": SIM_REPS, "seed": SIM_SEED,
            "n2": {"power": emp2, "mc_se": se2},
            f"n{n80}": {"power": empn, "mc_se": sen,
                        "analytic_t": power_at_n80},
        }

    # --- outputs -----------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "config": {
            "margin": MARGIN, "alpha": ALPHA,
            "sgld_runs": [stem for _, stem in SGLD_RUNS],
            "pc_runs": [stem for _, stem in PC_RUNS],
            "m0": m0, "m1": m1,
        },
        "per_run": {r["label"]: {"stem": r["stem"], "auroc": r["auroc"],
                                 "macro5": r["macro5"],
                                 "v_macro": v_macro_runs[r["label"]],
                                 "v_per_ds": v_per_ds_runs[r["label"]]}
                    for r in all_runs},
        "seed_level": {
            "sgld_macro5": sgld_macro.tolist(), "pc_macro5": pc_macro.tolist(),
            "s2_sgld": s2_sgld, "s2_pc": s2_pc, "s2_pooled": s2_pooled,
            "s_pooled": s_pooled, "v_bar_macro": v_bar_macro,
            "v_bar_share_of_s2": v_bar_macro / s2_pooled,
        },
        "score_correlation": rho,
        "tost": asdict(tost),
        "published_hierarchical_ci": (
            [published["hierarchical_bootstrap"]["macro5"]["ci_low"],
             published["hierarchical_bootstrap"]["macro5"]["ci_high"]]
            if published else None),
        "retrospective_power": retro,
        "design_table": {str(m): {str(pw): n for pw, n in row.items()}
                         for m, row in design.items()},
        "n_star_interval_80": {"df": df_s2, "s2_lo": s2_lo, "s2_hi": s2_hi,
                               "n_lo": n80_lo, "n_hi": n80_hi},
        "empirical_power": empirical,
    }
    json_path = OUT_DIR / "case_study.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {json_path}")

    md = ["# Case study: the four training runs through the equivalence protocol\n\n"]
    md.append(f"Margin {MARGIN}, alpha {ALPHA}. ID: CIFAR-10 test (m0={m0}); OOD sizes: "
              + ", ".join(f"{ds} {m1[ds]}" for ds in DATASETS) + ".\n\n")
    md.append("## Per-run AUROCs\n\n")
    md.append("| Run | " + " | ".join(ds.upper() for ds in DATASETS) + " | macro-5 |\n")
    md.append("|---|" + "---:|" * (len(DATASETS) + 1) + "\n")
    for r in all_runs:
        md.append(f"| {r['label']} | "
                  + " | ".join(f"{r['auroc'][ds]:.4f}" for ds in DATASETS)
                  + f" | **{r['macro5']:.4f}** |\n")
    md.append(f"\nSGLD mean {sgld_macro.mean():.4f}, PC mean {pc_macro.mean():.4f}, "
              f"delta (PC - SGLD) = {tost.delta:+.4f}.\n\n")
    md.append("## Estimates\n\n")
    md.append(f"- s_SGLD = {np.sqrt(s2_sgld):.4f}, s_PC = {np.sqrt(s2_pc):.4f}, "
              f"pooled s = {s_pooled:.4f} (s^2 = {s2_pooled:.3e})\n")
    md.append(f"- v_bar(macro-5) = {v_bar_macro:.3e} = {v_bar_macro / s2_pooled:.1%} of s^2\n")
    md.append(f"- between-run score correlation: "
              + ", ".join(f"{k} {c['mean']:.2f}" for k, c in rho.items()) + "\n\n")
    md.append("## TOST (reproduced)\n\n")
    md.append(f"- delta = {tost.delta:+.4f}, SE = {tost.se:.4f}, df = {tost.df:.2f}, "
              f"p = {tost.p_tost:.4f} -> no equivalence at margin {MARGIN}\n")
    md.append(f"- Welch 90% CI [{tost.ci90[0]:+.4f}, {tost.ci90[1]:+.4f}]"
              + (f"; published hierarchical 95% CI "
                 f"[{payload['published_hierarchical_ci'][0]:+.4f}, "
                 f"{payload['published_hierarchical_ci'][1]:+.4f}]\n" if published else "\n"))
    md.append("\n## Power\n\n")
    md.append(f"- retrospective (n=2, delta=0): t {retro['power_t_delta0']:.4f}, "
              f"normal {retro['power_normal_delta0']:.4f}\n")
    if empirical:
        md.append(f"- empirical (calibrated proxy): n=2 -> {empirical['n2']['power']:.3f} "
                  f"(MC SE {empirical['n2']['mc_se']:.3f}); n={n80} -> "
                  f"{empirical[f'n{n80}']['power']:.3f} (MC SE {empirical[f'n{n80}']['mc_se']:.3f})\n")
    md.append(f"- n*(delta=0.01) = {n80} (80%), {n90} (90%); "
              f"chi^2 95% interval on n*(80%): [{n80_lo}, {n80_hi}] (df = {df_s2:.2f})\n\n")
    md.append("## Design table n*(margin, power)\n\n")
    md.append("| margin | " + " | ".join(f"{int(pw * 100)}%" for pw in POWER_TARGETS) + " |\n")
    md.append("|---:|" + "---:|" * len(POWER_TARGETS) + "\n")
    for margin in DESIGN_MARGINS:
        md.append(f"| {margin} | "
                  + " | ".join(str(design[margin][pw]) for pw in POWER_TARGETS) + " |\n")
    md_path = OUT_DIR / "case_study.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
