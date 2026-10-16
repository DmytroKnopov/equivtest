"""case2_multiberts.py -- the equivalence protocol on MultiBERTs GLUE predictions.

Second case study of the accompanying article, MultiBERTs part, on the
per-example dev-set predictions released with MultiBERTs (Sellam et al.,
ICLR 2022): for RTE, SST-2 and MNLI (matched dev set) all 620 prediction
files per task are downloaded, per-run accuracy and per-item correctness
are computed from argmax(probabilities) against the released labels, and

  1. the two-level summary of Analysis A of case2_modelzoo.py is computed for
     the configuration release=multiberts, 2M steps, lr 2e-5, fine-tuning
     seed 0 (25 independent pretraining seeds): s, v_bar = mean p(1-p)/m,
     v_bar/s^2, between-run correlation of correctness, item agreement,
     design table;
  2. a nested aside splits the accuracy variance of all 5 x 25 runs of that
     configuration into between-pretraining-seed and within (fine-tuning)
     components (one-way random-effects ANOVA, method of moments);
  3. comparison (i): lr 2e-5 vs 5e-5 at 2M steps, 25 vs 25 runs -- full-data
     Welch-TOST, certification rate vs n by disjoint subsampling, formula,
     pilot-then-design;
  4. comparison (ii): 1M vs 2M pretraining steps for pretraining seeds 0-4,
     5 vs 5 -- unpaired Welch-TOST as the protocol prescribes and a paired
     one-sample TOST as a sensitivity check; n* implied by the pooled s;
  5. a null-true check on the real seed law: disjoint arms from the 25 runs,
     Delta = 0 and Delta = +-margin by shift.

Outputs: results/case2_multiberts.json, case2_multiberts.md,
case2_runs_multiberts.csv; data/derived/multiberts_<task>_correct.npy. The
downloaded prediction files (about 230 MB) are kept in data/multiberts/.

Usage (from repo root):
    python experiments/case2_multiberts.py [--fast]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from scipy.stats import f as f_dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from equivtest.accuracy import between_run_correlation, item_agreement  # noqa: E402
from equivtest.power import seeds_needed  # noqa: E402
from case2_common import (  # noqa: E402
    ALPHA, POWER_TARGETS, ROOT_SEED, design_table, fmt_rate, moment_summary,
    null_true_split, paired_tost, pilot_then_design, software_versions, today,
    tost_dict, two_arm_subsampling,
)

BASE_URL = "https://storage.googleapis.com/multiberts/public/example-predictions/GLUE/"
DATA_DIR = REPO / "data" / "multiberts"
DERIVED_DIR = REPO / "data" / "derived"
OUT_DIR = REPO / "results"

TASKS = ("RTE", "SST-2", "MNLI")
EXPECTED_ROWS = {"RTE": 277, "SST-2": 872, "MNLI": 9815}
MARGINS = (0.01, 0.02)
N_GRID = (2, 3, 5, 8, 12)
REPS_FULL = 2000
REPS_FAST = 200
PILOT_N0 = 5

SINGLE = {"release": "multiberts", "n_steps": "2M", "lr": "2e-05", "ft_seed": "0"}
COMPARISON_I = ("2e-05", "5e-05")     # lr A vs lr B at 2M, ft_seed 0
COMPARISON_II = ("1M", "2M")          # n_steps A vs B, pretrain ids 0-4, lr 2e-05, ft_seed 0


# --- data -----------------------------------------------------------------------

def fetch(session: requests.Session, rel: str) -> tuple[str, str, int]:
    dest = DATA_DIR / rel
    if dest.exists() and dest.stat().st_size > 0:
        return rel, "present", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(6):
        try:
            r = session.get(BASE_URL + rel, timeout=120)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return rel, "downloaded", len(r.content)
        except requests.RequestException as exc:
            err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{rel}: {err}")


def ensure_files(rels: list[str], workers: int = 8) -> dict:
    counts = {"present": 0, "downloaded": 0}
    total = 0
    t0 = time.perf_counter()
    with requests.Session() as s, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch, s, rel) for rel in rels]
        for i, fut in enumerate(as_completed(futures), 1):
            _, status, size = fut.result()
            counts[status] += 1
            total += size
            if i % 500 == 0:
                print(f"    [{i}/{len(rels)}] {time.perf_counter() - t0:.0f}s", flush=True)
    counts["bytes"] = total
    counts["seconds"] = time.perf_counter() - t0
    return counts


def read_catalogue() -> list[dict]:
    with requests.Session() as s:
        fetch(s, "index.tsv")
    with open(DATA_DIR / "index.tsv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return rows


def parse_predictions(path: Path, n_classes: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    arr = np.array(text.split(), dtype=np.float64)
    return arr.reshape(-1, n_classes)


def build_task(task: str, rows: list[dict]) -> dict:
    labels = np.array((DATA_DIR / f"{task}_dev_labels").read_text(encoding="utf-8").split(),
                      dtype=np.int64)
    m = len(labels)
    files = sorted(r["file"] for r in rows)
    npy = DERIVED_DIR / f"multiberts_{task}_correct.npy"
    idx_csv = DERIVED_DIR / f"multiberts_{task}_correct_index.csv"
    if npy.exists() and idx_csv.exists():
        with open(idx_csv, encoding="utf-8", newline="") as fh:
            cached = [r["file"] for r in csv.DictReader(fh)]
        if cached == files:
            correct = np.load(npy)
            if correct.shape == (len(files), m):
                print(f"  {task}: loaded cached correctness {correct.shape}")
                return {"labels": labels, "files": files, "correct": correct, "m": m,
                        "n_classes": int(labels.max()) + 1, "label_counts": np.bincount(labels).tolist()}
    sample = (DATA_DIR / files[0]).read_text(encoding="utf-8").splitlines()
    n_classes = len(sample[0].split("\t"))
    correct = np.empty((len(files), m), dtype=np.uint8)
    t0 = time.perf_counter()
    for i, rel in enumerate(files):
        probs = parse_predictions(DATA_DIR / rel, n_classes)
        if probs.shape[0] != m:
            raise RuntimeError(f"{rel}: {probs.shape[0]} rows, expected {m}")
        correct[i] = (probs.argmax(axis=1) == labels).astype(np.uint8)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, correct)
    with open(idx_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "file"])
        for i, rel in enumerate(files):
            w.writerow([i, rel])
    print(f"  {task}: parsed {len(files)} files x {m} items ({n_classes} classes) in "
          f"{time.perf_counter() - t0:.0f}s; label counts {np.bincount(labels).tolist()}")
    return {"labels": labels, "files": files, "correct": correct, "m": m,
            "n_classes": n_classes, "label_counts": np.bincount(labels).tolist()}


def select(rows: list[dict], **conds) -> list[dict]:
    out = [r for r in rows if all(r[k] == v for k, v in conds.items())]
    return sorted(out, key=lambda r: (int(r["pretrain_id"]), int(r["ft_seed"])))


# --- nested variance split ------------------------------------------------------

def variance_components(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects ANOVA, method of moments.

    sigma2_within = MS_within; sigma2_between = (MS_between - MS_within)/n0 with
    n0 = (N - sum n_i^2 / N)/(a - 1) (equals the group size when balanced),
    truncated at zero.
    """
    values = np.asarray(values, dtype=np.float64)
    ids, inv = np.unique(groups, return_inverse=True)
    a = len(ids)
    n_i = np.bincount(inv)
    n_total = len(values)
    grand = values.mean()
    means = np.array([values[inv == g].mean() for g in range(a)])
    ss_between = float((n_i * (means - grand) ** 2).sum())
    ss_within = float(((values - means[inv]) ** 2).sum())
    ms_between = ss_between / (a - 1)
    ms_within = ss_within / (n_total - a)
    n0 = (n_total - (n_i**2).sum() / n_total) / (a - 1)
    s2_between = max((ms_between - ms_within) / n0, 0.0)
    s2_within = ms_within
    f_stat = ms_between / ms_within
    return {
        "n_groups": int(a), "group_sizes": sorted(set(n_i.tolist())), "n_total": int(n_total),
        "grand_mean": float(grand), "ms_between": ms_between, "ms_within": ms_within,
        "sigma2_between_pretraining": s2_between, "sigma2_within_finetuning": s2_within,
        "sd_between_pretraining": float(np.sqrt(s2_between)), "sd_within_finetuning": float(np.sqrt(s2_within)),
        "ratio_between_over_within": s2_between / s2_within if s2_within > 0 else None,
        "icc": s2_between / (s2_between + s2_within) if (s2_between + s2_within) > 0 else None,
        "F": f_stat, "p_value": float(1.0 - f_dist.cdf(f_stat, a - 1, n_total - a)),
        "total_variance_of_single_runs": float(values.var(ddof=1)),
    }


# --- main -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="smoke test: 200 replications")
    args = parser.parse_args()
    reps = REPS_FAST if args.fast else REPS_FULL
    t_start = time.perf_counter()

    print("Catalogue and downloads...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalogue = read_catalogue()
    rows_by_task = {t: [r for r in catalogue if r["task"] == t] for t in TASKS}
    rels = [f"{t}_dev_labels" for t in TASKS] + [r["file"] for t in TASKS for r in rows_by_task[t]]
    dl = ensure_files(rels)
    print(f"  {len(catalogue)} catalogue rows; {len(rels)} files for {TASKS}: "
          f"{dl['downloaded']} downloaded, {dl['present']} present, {dl['bytes'] / 1e6:.1f} MB, {dl['seconds']:.0f}s")

    print("\nParsing predictions...")
    tasks = {}
    for t in TASKS:
        tasks[t] = build_task(t, rows_by_task[t])
        if tasks[t]["m"] != EXPECTED_ROWS[t]:
            print(f"  WARNING: {t} has {tasks[t]['m']} rows, expected {EXPECTED_ROWS[t]}")

    # per-run table
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / f"case2_runs_multiberts{'_fast' if args.fast else ''}.csv"
    file_row = {t: {rel: i for i, rel in enumerate(tasks[t]["files"])} for t in TASKS}
    acc_of = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "release", "pretrain_id", "n_steps", "lr", "ft_seed", "accuracy", "file"])
        for t in TASKS:
            acc_all = tasks[t]["correct"].mean(axis=1)
            for r in sorted(rows_by_task[t], key=lambda r: (r["release"], r["n_steps"], r["lr"],
                                                             int(r["pretrain_id"]), int(r["ft_seed"]))):
                a = float(acc_all[file_row[t][r["file"]]])
                acc_of[(t, r["file"])] = a
                w.writerow([t, r["release"], r["pretrain_id"], r["n_steps"], r["lr"], r["ft_seed"],
                            f"{a:.6f}", r["file"]])
    print(f"wrote {csv_path}")

    def accs(t: str, rows: list[dict]) -> np.ndarray:
        return np.array([acc_of[(t, r["file"])] for r in rows])

    def matrix(t: str, rows: list[dict]) -> np.ndarray:
        return tasks[t]["correct"][[file_row[t][r["file"]] for r in rows]]

    rng = np.random.default_rng(ROOT_SEED)
    results = {}
    for t in TASKS:
        print(f"\n===== {t} (m = {tasks[t]['m']}) =====")
        m = tasks[t]["m"]
        res: dict = {"m": m, "n_classes": tasks[t]["n_classes"], "label_counts": tasks[t]["label_counts"],
                     "n_files": len(tasks[t]["files"])}

        # --- single configuration: 25 pretraining seeds -----------------------------
        rows25 = select(rows_by_task[t], **SINGLE)
        acc25 = accs(t, rows25)
        mat25 = matrix(t, rows25)
        summary = moment_summary(acc25, m, rng)
        s = summary["sd"]
        rho = between_run_correlation(mat25)
        agreement = item_agreement(mat25)
        design = design_table(s, MARGINS)
        res["single_configuration"] = {
            "config": SINGLE, "n_runs": len(rows25), "pretrain_ids": [int(r["pretrain_id"]) for r in rows25],
            "accuracies": acc25.tolist(), "summary": summary, "rho": rho,
            "item_agreement": agreement, "design_table": design,
        }
        print(f"  single configuration ({len(rows25)} runs): mean {summary['mean']:.4f}, s = {s:.5f}, "
              f"skew {summary['skewness']:+.3f}, exc. kurtosis {summary['excess_kurtosis']:+.3f}, "
              f"Shapiro p = {summary['shapiro']['p']:.3g}")
        print(f"  v_bar = {summary['v_bar']:.3e}, s^2 = {summary['s2']:.3e}, v_bar / s^2 = {summary['v_bar_over_s2']:.3f}, "
              f"rho = {rho:.4f}; items always/never/mixed = {agreement['always_correct']:.3f}/"
              f"{agreement['never_correct']:.3f}/{agreement['mixed']:.3f}")
        print("  design table: " + "; ".join(
            f"margin {d}: " + ", ".join(f"{int(pw * 100)}% -> {design[str(d)][str(pw)]}" for pw in POWER_TARGETS)
            for d in MARGINS))

        # --- nested aside ------------------------------------------------------------
        rows125 = select(rows_by_task[t], release="multiberts", n_steps="2M", lr="2e-05")
        vc = variance_components(accs(t, rows125), np.array([int(r["pretrain_id"]) for r in rows125]))
        res["nested_variance"] = vc
        print(f"  nested ({vc['n_total']} runs, {vc['n_groups']} pretraining seeds x {vc['group_sizes']} fine-tuning seeds): "
              f"sd between pretraining {vc['sd_between_pretraining']:.5f}, sd within fine-tuning {vc['sd_within_finetuning']:.5f}, "
              f"ratio of variances {vc['ratio_between_over_within']:.3f}, ICC {vc['icc']:.3f}, F p = {vc['p_value']:.3g}")

        # --- comparison (i): lr 2e-5 vs 5e-5 -----------------------------------------
        rows_a = select(rows_by_task[t], release="multiberts", n_steps="2M", lr=COMPARISON_I[0], ft_seed="0")
        rows_b = select(rows_by_task[t], release="multiberts", n_steps="2M", lr=COMPARISON_I[1], ft_seed="0")
        acc_a, acc_b = accs(t, rows_a), accs(t, rows_b)
        cross = between_run_correlation(np.vstack([matrix(t, rows_a), matrix(t, rows_b)]))
        t0 = time.perf_counter()
        sub = two_arm_subsampling(acc_a, acc_b, N_GRID, MARGINS, reps, rng)
        pilot = pilot_then_design(acc_a, acc_b, PILOT_N0, MARGINS, reps, rng)
        comp_i = {
            "arms": {"a": f"lr={COMPARISON_I[0]}", "b": f"lr={COMPARISON_I[1]}"},
            "n_a": len(rows_a), "n_b": len(rows_b), "acc_a": acc_a.tolist(), "acc_b": acc_b.tolist(),
            "mean_a": float(acc_a.mean()), "mean_b": float(acc_b.mean()),
            "sd_a": float(acc_a.std(ddof=1)), "sd_b": float(acc_b.std(ddof=1)),
            "rho_pooled_runs": cross,
            "full_data_tost": {str(d): tost_dict(acc_a, acc_b, d) for d in MARGINS},
            "subsampling": sub, "pilot_then_design": pilot,
        }
        res["comparison_i_lr"] = comp_i
        print(f"  comparison (i) lr {COMPARISON_I[0]} vs {COMPARISON_I[1]}: mean A {acc_a.mean():.4f} (sd {acc_a.std(ddof=1):.5f}), "
              f"mean B {acc_b.mean():.4f} (sd {acc_b.std(ddof=1):.5f}), pooled s {sub['s_pooled']:.5f} [{time.perf_counter() - t0:.0f}s]")
        for d in MARGINS:
            f = comp_i["full_data_tost"][str(d)]
            rates = ", ".join(f"n={n}: {fmt_rate(sub['by_n'][str(n)][str(d)]['certification_rate'], sub['by_n'][str(n)][str(d)]['mc_se'])}"
                              for n in N_GRID)
            pl = pilot["by_margin"][str(d)]
            print(f"    margin {d}: delta {f['delta']:+.4f}, 90% CI [{f['ci90'][0]:+.4f}, {f['ci90'][1]:+.4f}], p = {f['p_tost']:.4f}, "
                  f"equivalent {f['equivalent']}; n*(80%) from full s = {pilot['n_star_full'][str(d)]}")
            print(f"      rates: {rates}; pilot n* median {pl['median']:.0f} [{pl['p2.5']:.0f}, {pl['p97.5']:.0f}]")

        # --- comparison (ii): 1M vs 2M, pretraining seeds 0-4 -----------------------
        rows_1m = [r for r in select(rows_by_task[t], release="multiberts", n_steps=COMPARISON_II[0], lr="2e-05", ft_seed="0")
                   if int(r["pretrain_id"]) <= 4]
        rows_2m = [r for r in select(rows_by_task[t], release="multiberts", n_steps=COMPARISON_II[1], lr="2e-05", ft_seed="0")
                   if int(r["pretrain_id"]) <= 4]
        acc_1m, acc_2m = accs(t, rows_1m), accs(t, rows_2m)
        assert [r["pretrain_id"] for r in rows_1m] == [r["pretrain_id"] for r in rows_2m]
        s_pooled_ii = float(np.sqrt((acc_1m.var(ddof=1) + acc_2m.var(ddof=1)) / 2.0))
        comp_ii = {
            "arms": {"a": f"n_steps={COMPARISON_II[0]}", "b": f"n_steps={COMPARISON_II[1]}"},
            "pretrain_ids": [int(r["pretrain_id"]) for r in rows_1m],
            "acc_a": acc_1m.tolist(), "acc_b": acc_2m.tolist(),
            "mean_a": float(acc_1m.mean()), "mean_b": float(acc_2m.mean()),
            "sd_a": float(acc_1m.std(ddof=1)), "sd_b": float(acc_2m.std(ddof=1)),
            "s_pooled": s_pooled_ii,
            "sd_paired_differences": float((acc_1m - acc_2m).std(ddof=1)),
            "unpaired_tost": {str(d): tost_dict(acc_1m, acc_2m, d) for d in MARGINS},
            "paired_tost": {str(d): paired_tost(acc_1m - acc_2m, d) for d in MARGINS},
            "n_star_from_pooled_s": {str(d): {str(pw): int(seeds_needed(d, pw, s_pooled_ii, 0.0, ALPHA))
                                              for pw in POWER_TARGETS} for d in MARGINS},
        }
        res["comparison_ii_steps"] = comp_ii
        print(f"  comparison (ii) {COMPARISON_II[0]} vs {COMPARISON_II[1]} (pretraining seeds 0-4): mean A {acc_1m.mean():.4f}, "
              f"mean B {acc_2m.mean():.4f}, pooled s {s_pooled_ii:.5f}, sd of paired differences {comp_ii['sd_paired_differences']:.5f}")
        for d in MARGINS:
            u, p = comp_ii["unpaired_tost"][str(d)], comp_ii["paired_tost"][str(d)]
            print(f"    margin {d}: unpaired delta {u['delta']:+.4f}, CI [{u['ci90'][0]:+.4f}, {u['ci90'][1]:+.4f}], p = {u['p_tost']:.4f}, "
                  f"eq {u['equivalent']}; paired CI [{p['ci90'][0]:+.4f}, {p['ci90'][1]:+.4f}], p = {p['p_tost']:.4f}, eq {p['equivalent']}; "
                  f"n*(80%/90%) = {comp_ii['n_star_from_pooled_s'][str(d)]['0.8']}/{comp_ii['n_star_from_pooled_s'][str(d)]['0.9']}")

        # --- null-true check ---------------------------------------------------------
        t0 = time.perf_counter()
        null = null_true_split(acc25, N_GRID, MARGINS, reps, rng)
        null["margins"] = list(MARGINS)
        null["n_star_formula"] = {str(d): {str(pw): int(seeds_needed(d, pw, s, 0.0, ALPHA)) for pw in POWER_TARGETS}
                                  for d in MARGINS}
        res["null_true_check"] = null
        print(f"  null-true check ({reps} replications, {time.perf_counter() - t0:.0f}s):")
        for d in MARGINS:
            key = str(d)
            print(f"    margin {d} (n*(80%) = {null['n_star_formula'][key]['0.8']}): " + "; ".join(
                f"n={n}: {fmt_rate(null['by_n'][str(n)][key]['certification_rate_delta0'], null['by_n'][str(n)][key]['mc_se_delta0'])} "
                f"[t {null['by_n'][str(n)][key]['formula']['t']:.3f}] size {null['by_n'][str(n)][key]['size_at_plus_margin']:.3f}/"
                f"{null['by_n'][str(n)][key]['size_at_minus_margin']:.3f}"
                for n in N_GRID if "skipped" not in null["by_n"][str(n)]))
        results[t] = res

    # --- outputs -----------------------------------------------------------------
    runtime = time.perf_counter() - t_start
    payload = {
        "provenance": {
            "date": today(), "root_seed": ROOT_SEED, "fast": args.fast, "runtime_seconds": runtime,
            "software": software_versions(),
            "base_url": BASE_URL, "catalogue": BASE_URL + "index.tsv",
            "label_files": {t: BASE_URL + f"{t}_dev_labels" for t in TASKS},
            "n_files_per_task": {t: len(tasks[t]["files"]) for t in TASKS},
            "download": {k: dl[k] for k in ("downloaded", "present", "bytes")},
            "rows_per_task": {t: tasks[t]["m"] for t in TASKS},
            "metric": "accuracy = mean(argmax(probabilities) == label) over the dev set",
            "sign_convention": "delta = mean(arm a) - mean(arm b) as named in 'arms'",
            "reps": reps, "n_grid": list(N_GRID), "margins": list(MARGINS), "pilot_n0": PILOT_N0,
        },
        "tasks": results,
    }
    suffix = "_smoke" if args.fast else ""
    json_path = OUT_DIR / f"case2_multiberts{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {json_path}")

    md = ["# Case study 2: MultiBERTs GLUE dev-set predictions (Sellam et al., 2022)\n\n"]
    md.append(f"Date {today()}, root seed {ROOT_SEED}, {reps} replications, runtime {runtime / 60:.1f} min"
              f"{' (FAST smoke run)' if args.fast else ''}. Accuracy = argmax of the released class probabilities "
              f"against the released dev labels. Sign convention: delta = mean(A) - mean(B).\n\n")
    md.append("## Single configuration: release=multiberts, 2M steps, lr 2e-5, fine-tuning seed 0 (25 pretraining seeds)\n\n")
    md.append("| task | m | mean | s | skewness | excess kurtosis | Shapiro p | v_bar | v_bar / s^2 | rho | always / never / mixed | n* 0.01 (80/90%) | n* 0.02 (80/90%) |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n")
    for t in TASKS:
        sc = results[t]["single_configuration"]
        su = sc["summary"]
        ag = sc["item_agreement"]
        md.append(f"| {t} | {results[t]['m']} | {su['mean']:.4f} | {su['sd']:.5f} | {su['skewness']:+.3f} | {su['excess_kurtosis']:+.3f} | "
                  f"{su['shapiro']['p']:.3g} | {su['v_bar']:.3e} | {su['v_bar_over_s2']:.3f} | {sc['rho']:.3f} | "
                  f"{ag['always_correct']:.3f} / {ag['never_correct']:.3f} / {ag['mixed']:.3f} | "
                  f"{sc['design_table']['0.01']['0.8']} / {sc['design_table']['0.01']['0.9']} | "
                  f"{sc['design_table']['0.02']['0.8']} / {sc['design_table']['0.02']['0.9']} |\n")
    md.append("\n## Nested variance split (all 5 fine-tuning seeds x 25 pretraining seeds, 2M steps, lr 2e-5)\n\n")
    md.append("| task | runs | sd between pretraining | sd within fine-tuning | variance ratio (between/within) | ICC | F p-value |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for t in TASKS:
        vc = results[t]["nested_variance"]
        md.append(f"| {t} | {vc['n_total']} | {vc['sd_between_pretraining']:.5f} | {vc['sd_within_finetuning']:.5f} | "
                  f"{vc['ratio_between_over_within']:.3f} | {vc['icc']:.3f} | {vc['p_value']:.3g} |\n")
    md.append(f"\n## Comparison (i): lr {COMPARISON_I[0]} vs {COMPARISON_I[1]} at 2M steps, fine-tuning seed 0 (25 vs 25)\n\n")
    md.append("| task | margin | delta | 90% CI | p | equivalent | pooled s | n*(80%) | " +
              " | ".join(f"rate n={n}" for n in N_GRID) + " | pilot n* median [2.5%, 97.5%] |\n|---|---:|---:|---|---:|---|---:|---:|" +
              "---:|" * len(N_GRID) + "---|\n")
    for t in TASKS:
        c = results[t]["comparison_i_lr"]
        for d in MARGINS:
            f = c["full_data_tost"][str(d)]
            rates = " | ".join(fmt_rate(c["subsampling"]["by_n"][str(n)][str(d)]["certification_rate"],
                                        c["subsampling"]["by_n"][str(n)][str(d)]["mc_se"]) for n in N_GRID)
            pl = c["pilot_then_design"]["by_margin"][str(d)]
            md.append(f"| {t} | {d} | {f['delta']:+.4f} | [{f['ci90'][0]:+.4f}, {f['ci90'][1]:+.4f}] | {f['p_tost']:.4f} | {f['equivalent']} | "
                      f"{c['subsampling']['s_pooled']:.5f} | {c['pilot_then_design']['n_star_full'][str(d)]} | {rates} | "
                      f"{pl['median']:.0f} [{pl['p2.5']:.0f}, {pl['p97.5']:.0f}] |\n")
    md.append(f"\n## Comparison (ii): {COMPARISON_II[0]} vs {COMPARISON_II[1]} pretraining steps, pretraining seeds 0-4, lr 2e-5, fine-tuning seed 0 (5 vs 5)\n\n")
    md.append("| task | margin | delta | unpaired 90% CI | p | equivalent | paired 90% CI | paired p | paired equivalent | pooled s | n*(80/90%) |\n"
              "|---|---:|---:|---|---:|---|---|---:|---|---:|---|\n")
    for t in TASKS:
        c = results[t]["comparison_ii_steps"]
        for d in MARGINS:
            u, p = c["unpaired_tost"][str(d)], c["paired_tost"][str(d)]
            md.append(f"| {t} | {d} | {u['delta']:+.4f} | [{u['ci90'][0]:+.4f}, {u['ci90'][1]:+.4f}] | {u['p_tost']:.4f} | {u['equivalent']} | "
                      f"[{p['ci90'][0]:+.4f}, {p['ci90'][1]:+.4f}] | {p['p_tost']:.4f} | {p['equivalent']} | {c['s_pooled']:.5f} | "
                      f"{c['n_star_from_pooled_s'][str(d)]['0.8']} / {c['n_star_from_pooled_s'][str(d)]['0.9']} |\n")
    md.append("\n## Null-true check (disjoint arms of the 25 runs of the single configuration)\n\n")
    md.append("| task | margin | n | certification rate (Delta=0) | formula (t) | formula (normal) | size at +margin | size at -margin |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for t in TASKS:
        null = results[t]["null_true_check"]
        for d in MARGINS:
            for n in N_GRID:
                row = null["by_n"][str(n)]
                if "skipped" in row:
                    continue
                r = row[str(d)]
                md.append(f"| {t} | {d} | {n} | {fmt_rate(r['certification_rate_delta0'], r['mc_se_delta0'])} | {r['formula']['t']:.3f} | "
                          f"{r['formula']['normal']:.3f} | {fmt_rate(r['size_at_plus_margin'], r['mc_se_plus'])} | "
                          f"{fmt_rate(r['size_at_minus_margin'], r['mc_se_minus'])} |\n")
    md_path = OUT_DIR / f"case2_multiberts{suffix}.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"total runtime {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
