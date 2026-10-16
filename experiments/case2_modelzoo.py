"""case2_modelzoo.py -- the equivalence protocol on the CIFAR-10 Model Zoos.

Second case study of the accompanying article (Analyses A-C) on the CNN-s
zoos of Schuerholt et al. (NeurIPS 2022 Datasets and Benchmarks, Zenodo
record 6620869, CC BY 4.0):

  1. Data: the seed zoo (one configuration, seeds 1..1000) and the
     hyperparameter zoo with fixed seeds (1280 configurations x seeds). The
     zips are downloaded with resume, checksummed, and the per-trial
     params.json, result.json and last checkpoint are extracted. Per-image
     correctness on the 10,000 CIFAR-10 test images (fetched by torchvision
     into data/) is recomputed from the last checkpoint with the authors'
     test pipeline (Resize 28x28, ToTensor, Normalize 0.5/0.5) and checked
     against the shipped last-epoch test_acc.
  2. Analysis A: two-level structure of the 1000 seed-zoo accuracies (moments,
     binomial item-level variance, between-run correlation of correctness,
     item agreement, design table n*(margin, power)).
  3. Analysis B: real seed law -- disjoint arms of n runs from the seed zoo,
     TOST certification rate at Delta = 0 and size at Delta = +-margin
     against the formula, for margin 0.01 and a margin calibrated to n* = 12.
  4. Analysis C: two genuine two-method comparisons in the hyperparameter zoo
     (Adam vs SGD; GELU vs tanh; 10 seeds each): full-data TOST, small-n
     certification rates, pilot-then-design.

Outputs: results/case2_modelzoo.json, case2_modelzoo.md,
case2_runs_modelzoo.csv; data/derived/modelzoo_seed_correct.npy and
modelzoo_hypfix_correct.npy (R x 10000 uint8) with row indices. The Zenodo
archives (2.8 GB) are kept in data/modelzoo/.

Usage (from repo root, KMP_DUPLICATE_LIB_OK=TRUE):
    python experiments/case2_modelzoo.py [--fast] [--cpu]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import requests
import torch
import torchvision
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from equivtest.accuracy import between_run_correlation, item_agreement  # noqa: E402
from equivtest.power import seeds_needed  # noqa: E402
from case2_common import (  # noqa: E402
    ALPHA, POWER_TARGETS, ROOT_SEED, design_table, first_n_reaching, fmt_rate,
    moment_summary, null_true_split, pilot_then_design, sha256_file,
    software_versions, today, tost_dict, two_arm_subsampling,
)
from modelzoo_defs import CNN  # noqa: E402

DATA_DIR = REPO / "data" / "modelzoo"
DERIVED_DIR = REPO / "data" / "derived"
CIFAR_ROOT = REPO / "data"
OUT_DIR = REPO / "results"

ZENODO_RECORD = "https://zenodo.org/api/records/6620869"
ZIPS = {
    "seed": {"name": "cifar_small_seed.zip",
             "url": f"{ZENODO_RECORD}/files/cifar_small_seed.zip/content",
             "size": 898_696_417},
    "hypfix": {"name": "cifar_small_hyp_fix.zip",
               "url": f"{ZENODO_RECORD}/files/cifar_small_hyp_fix.zip/content",
               "size": 1_948_218_509},
}
MODEL_DEF_SOURCE = ("https://github.com/ModelZoos/ModelZooDataset/blob/"
                    "efccaf3fc2e15dc55788f132d2341861be13312e/code/model_definitions/def_net.py")

M_TEST = 10_000
TOLERANCE = 1e-3

MARGIN_1 = 0.01
DESIGN_MARGINS = (0.005, 0.01, 0.015, 0.02)
N_GRID_B = (2, 3, 5, 8, 12, 20, 50)
N_GRID_C = (2, 3, 5)
PAIR_MARGINS = (0.01, 0.02)
REPS_FULL = 5000
REPS_FAST = 200
PILOT_N0 = 5

# Hyperparameter zoo: the seed-zoo configuration (lr 1e-3 in that zoo) and the
# two nodes differing from it in exactly one factor.
BASE_NODE = {"model::init_type": "kaiming_uniform", "model::dropout": 0,
             "optim::wd": 0.01, "optim::optimizer": "adam", "model::nlin": "gelu"}
PAIRS = {
    "P1_adam_vs_sgd": ("optim::optimizer", "adam", "sgd"),
    "P2_gelu_vs_tanh": ("model::nlin", "gelu", "tanh"),
}


# --- data acquisition -----------------------------------------------------------

def download_with_resume(url: str, dest: Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  present: {dest.name} ({expected_size / 1e9:.2f} GB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= expected_size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        fh.write(chunk)
        except requests.RequestException as exc:
            print(f"  retry {attempt + 1}: {exc}")
            time.sleep(10)
    if dest.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: size {dest.stat().st_size} != {expected_size}")
    print(f"  downloaded: {dest.name}")


def safe_dir_name(name: str) -> str:
    return re.sub(r'[:<>"|?*]', "_", name)


def extract_needed(zip_path: Path, dest_root: Path) -> dict:
    """Extract params.json, result.json and the last checkpoint of every trial.

    Returns a summary of the archive layout; skipped when the marker exists.
    """
    marker = dest_root / f".{zip_path.stem}.extracted.json"
    if marker.exists():
        info = json.loads(marker.read_text(encoding="utf-8"))
        print(f"  extracted earlier: {info['top']} ({info['n_trials']} trials)")
        return info
    pat = re.compile(r"^([^/]+)/([^/]+)/checkpoint_(\d+)/checkpoints$")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        top = names[0].split("/")[0]
        ckpts: dict[str, list[int]] = collections.defaultdict(list)
        for n in names:
            m = pat.match(n)
            if m and m.group(1) == top:
                ckpts[m.group(2)].append(int(m.group(3)))
        experiment_files = sorted(n.split("/")[-1] for n in names
                                  if n.count("/") == 1 and not n.endswith("/"))
        sample_trial = sorted(ckpts)[0]
        trial_files = sorted(n.split("/")[-1] for n in names
                             if n.startswith(f"{top}/{sample_trial}/") and n.count("/") == 2
                             and not n.endswith("/"))
        last_ck = {t: max(v) for t, v in ckpts.items()}
        n_ck = collections.Counter(len(v) for v in ckpts.values())
        # Trial names of the hyperparameter zoo contain "::", which NTFS rejects.
        name_map = {safe_dir_name(t): t for t in ckpts}
        if len(name_map) != len(ckpts):
            raise RuntimeError("sanitised trial names collide")
        wanted = []
        for safe, t in name_map.items():
            last = last_ck[t]
            for rel in ("params.json", "result.json", f"checkpoint_{last:06d}/checkpoints"):
                wanted.append((f"{top}/{t}/{rel}", dest_root / top / safe / rel))
        t0 = time.perf_counter()
        for i, (member, out) in enumerate(wanted, 1):
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(member))
            if i % 600 == 0:
                print(f"    [{i}/{len(wanted)}] {time.perf_counter() - t0:.0f}s", flush=True)
    info = {"top": top, "n_trials": len(ckpts), "n_entries": len(names),
            "experiment_files": experiment_files, "trial_files": trial_files,
            "checkpoints_per_trial": {str(k): v for k, v in sorted(n_ck.items())},
            "last_checkpoint_index": sorted(set(last_ck.values())),
            "trial_name_map": name_map}
    marker.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"  extracted {len(wanted)} files from {zip_path.name}")
    return info


def print_layout(info: dict) -> None:
    print(f"  archive top-level dir: {info['top']}; {info['n_entries']} entries; "
          f"{info['n_trials']} Ray Tune trial dirs")
    print(f"  experiment-level files: {info['experiment_files']}")
    print(f"  trial-level files: {info['trial_files']}")
    print(f"  checkpoints per trial: {info['checkpoints_per_trial']}; "
          f"last checkpoint index: {info['last_checkpoint_index']}")


def load_trials(top_dir: Path, name_map: dict) -> list[dict]:
    trials = []
    for tdir in sorted(p for p in top_dir.iterdir() if p.is_dir()):
        archive_name = name_map.get(tdir.name, tdir.name)
        params = json.loads((tdir / "params.json").read_text(encoding="utf-8"))
        lines = [json.loads(l) for l in (tdir / "result.json").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        ck_dirs = sorted(p for p in tdir.glob("checkpoint_*") if (p / "checkpoints").exists())
        last_ck = ck_dirs[-1]
        ck_index = int(last_ck.name.split("_")[1])
        last_line = lines[-1]
        if last_line["training_iteration"] != ck_index:
            matching = [l for l in lines if l["training_iteration"] == ck_index]
            if not matching:
                raise RuntimeError(f"{tdir.name}: no result line for checkpoint {ck_index}")
            last_line = matching[-1]
        trials.append({
            "dir": tdir, "name": archive_name, "params": params, "seed": int(params["seed"]),
            "n_result_lines": len(lines), "checkpoint_index": ck_index,
            "checkpoint_path": last_ck / "checkpoints",
            "reported_test_acc": float(last_line["test_acc"]),
            "reported_train_acc": float(last_line.get("train_acc", float("nan"))),
        })
    return trials


def node_id(params: dict) -> str:
    return (f"init={params['model::init_type']},nlin={params['model::nlin']},"
            f"opt={params['optim::optimizer']},wd={params['optim::wd']},"
            f"dropout={params['model::dropout']},lr={params['optim::lr']}")


# --- inference ------------------------------------------------------------------

def load_test_set(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = torchvision.datasets.CIFAR10(root=str(CIFAR_ROOT), train=False, download=True,
                                      transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=2000, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x_all = torch.cat(xs).to(device)
    y_all = torch.cat(ys).to(device)
    assert x_all.shape == (M_TEST, 3, 28, 28), x_all.shape
    return x_all, y_all


def build_model(params: dict, device: torch.device) -> torch.nn.Module:
    model = CNN(channels_in=int(params["model::channels_in"]), nlin=params["model::nlin"],
                dropout=float(params["model::dropout"]), init_type=params["model::init_type"])
    return model.to(device).eval()


def load_state_dict(path: Path, device: torch.device, flags: dict) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        flags["weights_only_fallbacks"] = flags.get("weights_only_fallbacks", 0) + 1
        flags.setdefault("weights_only_error", repr(exc)[:300])
        return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def evaluate_trials(trials: list[dict], x_test: torch.Tensor, y_test: torch.Tensor,
                    device: torch.device, label: str) -> tuple[np.ndarray, dict]:
    correct = np.empty((len(trials), M_TEST), dtype=np.uint8)
    flags: dict = {}
    t0 = time.perf_counter()
    for i, tr in enumerate(trials):
        model = build_model(tr["params"], device)
        model.load_state_dict(load_state_dict(tr["checkpoint_path"], device, flags), strict=True)
        pred = model(x_test).argmax(dim=1)
        correct[i] = (pred == y_test).to(torch.uint8).cpu().numpy()
        tr["recomputed_acc"] = float(correct[i].mean())
        if (i + 1) % 200 == 0 or i + 1 == len(trials):
            print(f"    [{label}] {i + 1}/{len(trials)} runs, {time.perf_counter() - t0:.0f}s",
                  flush=True)
    flags["seconds"] = time.perf_counter() - t0
    return correct, flags


def sanity_report(trials: list[dict], label: str) -> dict:
    diff = np.array([tr["recomputed_acc"] - tr["reported_test_acc"] for tr in trials])
    absd = np.abs(diff)
    beyond = int((absd > TOLERANCE + 1e-12).sum())
    rep = {
        "n_runs": len(trials), "max_abs_diff": float(absd.max()),
        "mean_signed_diff": float(diff.mean()),
        "n_beyond_1e-3": beyond, "n_beyond_1e-4": int((absd > 1e-4 + 1e-12).sum()),
        "n_exact": int((absd < 1e-12).sum()),
        "worst_runs": [
            {"trial": trials[j]["name"], "reported": trials[j]["reported_test_acc"],
             "recomputed": trials[j]["recomputed_acc"]}
            for j in np.argsort(-absd)[:5]],
    }
    print(f"  [{label}] recomputed vs shipped test_acc: max |diff| = {rep['max_abs_diff']:.5f}, "
          f"{beyond} runs beyond {TOLERANCE}, {rep['n_beyond_1e-4']} beyond 1e-4, "
          f"{rep['n_exact']} exact, mean signed diff {rep['mean_signed_diff']:+.2e}")
    return rep


# --- margin calibration -----------------------------------------------------------

def calibrate_margin(s: float, target_n: int = 12, power: float = 0.8) -> dict:
    """Interval of margins with seeds_needed(margin, power, s, 0) == target_n."""
    coarse = np.arange(0.0005, 0.1, 0.0001)
    n_star = np.array([seeds_needed(float(d), power, s, 0.0, ALPHA) for d in coarse])
    hit = np.where(n_star == target_n)[0]
    if len(hit) == 0:
        raise RuntimeError(f"no margin on the coarse grid gives n* = {target_n}")
    lo_bracket = coarse[max(hit[0] - 1, 0)]
    hi_bracket = coarse[min(hit[-1] + 1, len(coarse) - 1)]
    fine = np.arange(lo_bracket, hi_bracket + 1e-9, 1e-6)
    n_fine = np.array([seeds_needed(float(d), power, s, 0.0, ALPHA) for d in fine])
    hit = np.where(n_fine == target_n)[0]
    lo, hi = float(fine[hit[0]]), float(fine[hit[-1]])
    mid = round((lo + hi) / 2.0, 4)
    if seeds_needed(mid, power, s, 0.0, ALPHA) != target_n:
        mid = (lo + hi) / 2.0
    return {"target_n": target_n, "power": power, "interval": [lo, hi], "margin": float(mid),
            "grid_step": 1e-6}


# --- main -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="smoke test: 60 seed-zoo runs, 200 replications")
    parser.add_argument("--cpu", action="store_true", help="run inference on the CPU")
    args = parser.parse_args()
    reps = REPS_FAST if args.fast else REPS_FULL
    t_start = time.perf_counter()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"device: {device}; torch {torch.__version__}; torchvision {torchvision.__version__}")

    # --- data --------------------------------------------------------------------
    print("\nDownloading / verifying the Zenodo archives...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    provenance_files = {}
    for key, spec in ZIPS.items():
        dest = DATA_DIR / spec["name"]
        download_with_resume(spec["url"], dest, spec["size"])
        t0 = time.perf_counter()
        digest = sha256_file(dest)
        provenance_files[key] = {"file": spec["name"], "url": spec["url"], "bytes": spec["size"],
                                 "sha256": digest}
        print(f"  sha256 {spec['name']} = {digest} ({time.perf_counter() - t0:.0f}s)")

    print("\nExtracting params.json, result.json and the last checkpoint of each trial...")
    layouts = {}
    for key, spec in ZIPS.items():
        layouts[key] = extract_needed(DATA_DIR / spec["name"], DATA_DIR)
        print_layout(layouts[key])

    print("\nReading trial metadata...")
    seed_trials = load_trials(DATA_DIR / layouts["seed"]["top"], layouts["seed"]["trial_name_map"])
    hyp_trials = load_trials(DATA_DIR / layouts["hypfix"]["top"], layouts["hypfix"]["trial_name_map"])
    seed_trials.sort(key=lambda t: t["seed"])
    print(f"  seed zoo: {len(seed_trials)} trials, seeds {seed_trials[0]['seed']}..{seed_trials[-1]['seed']}, "
          f"result lines per trial {sorted({t['n_result_lines'] for t in seed_trials})}, "
          f"last checkpoint {sorted({t['checkpoint_index'] for t in seed_trials})}")
    seed_configs = {node_id(t["params"]) for t in seed_trials}
    print(f"  seed zoo configurations: {seed_configs}")
    nodes = collections.defaultdict(list)
    for t in hyp_trials:
        nodes[node_id(t["params"])].append(t)
    print(f"  hyperparameter zoo: {len(hyp_trials)} trials in {len(nodes)} nodes, "
          f"seeds per node {sorted({len(v) for v in nodes.values()})}")
    if args.fast:
        seed_trials = seed_trials[:60]

    def select_node(overrides: dict) -> list[dict]:
        spec = dict(BASE_NODE)
        spec.update(overrides)
        found = [t for t in hyp_trials
                 if all(t["params"][k] == v for k, v in spec.items())]
        found.sort(key=lambda t: t["seed"])
        return found

    pair_nodes = {}
    for pname, (factor, level_a, level_b) in PAIRS.items():
        a = select_node({factor: level_a})
        b = select_node({factor: level_b})
        pair_nodes[pname] = (a, b)
        print(f"  {pname}: {node_id(a[0]['params'])} ({len(a)} seeds) vs "
              f"{node_id(b[0]['params'])} ({len(b)} seeds)")
    hyp_selected = {}
    for a, b in pair_nodes.values():
        for t in a + b:
            hyp_selected[t["name"]] = t
    hyp_selected = sorted(hyp_selected.values(), key=lambda t: (node_id(t["params"]), t["seed"]))
    print(f"  distinct hyperparameter-zoo runs to evaluate: {len(hyp_selected)}")

    # --- inference ----------------------------------------------------------------
    print("\nLoading the CIFAR-10 test set (Resize 28x28, ToTensor, Normalize 0.5)...")
    x_test, y_test = load_test_set(device)
    print(f"  {tuple(x_test.shape)} on {device}; label counts {np.bincount(y_test.cpu().numpy()).tolist()}")

    print("\nRecomputing per-image correctness from the last checkpoints...")
    seed_correct, seed_flags = evaluate_trials(seed_trials, x_test, y_test, device, "seed zoo")
    hyp_correct, hyp_flags = evaluate_trials(hyp_selected, x_test, y_test, device, "hyp zoo")
    for flags in (seed_flags, hyp_flags):
        if flags.get("weights_only_fallbacks"):
            print(f"  NOTE: torch.load fell back to weights_only=False for "
                  f"{flags['weights_only_fallbacks']} checkpoints: {flags['weights_only_error']}")
    sanity = {"seed_zoo": sanity_report(seed_trials, "seed zoo"),
              "hyp_zoo": sanity_report(hyp_selected, "hyp zoo")}

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.fast else ""
    np.save(DERIVED_DIR / f"modelzoo_seed_correct{suffix}.npy", seed_correct)
    np.save(DERIVED_DIR / f"modelzoo_hypfix_correct{suffix}.npy", hyp_correct)
    with open(DERIVED_DIR / f"modelzoo_hypfix_correct_index{suffix}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "node", "seed", "trial"])
        for i, t in enumerate(hyp_selected):
            w.writerow([i, node_id(t["params"]), t["seed"], t["name"]])
    with open(DERIVED_DIR / f"modelzoo_seed_correct_index{suffix}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "seed", "trial"])
        for i, t in enumerate(seed_trials):
            w.writerow([i, t["seed"], t["name"]])

    # --- Analysis A --------------------------------------------------------------
    rng = np.random.default_rng(ROOT_SEED)
    acc = seed_correct.mean(axis=1)
    print("\nAnalysis A: two-level structure of the seed zoo")
    summary = moment_summary(acc, M_TEST, rng)
    s = summary["sd"]
    rho_full = between_run_correlation(seed_correct)
    sub_idx = rng.choice(len(acc), size=min(200, len(acc)), replace=False)
    rho_sub200 = between_run_correlation(seed_correct[sub_idx])
    agreement = item_agreement(seed_correct)
    design = design_table(s, DESIGN_MARGINS)
    print(f"  R = {summary['n_runs']}: mean {summary['mean']:.4f}, s = {s:.5f}, "
          f"skew {summary['skewness']:+.3f}, excess kurtosis {summary['excess_kurtosis']:+.3f}, "
          f"range [{summary['min']:.4f}, {summary['max']:.4f}]")
    print(f"  Shapiro-Wilk (n = {summary['shapiro']['n']}): W = {summary['shapiro']['W']:.4f}, "
          f"p = {summary['shapiro']['p']:.3g}")
    print(f"  v_bar = {summary['v_bar']:.3e}, s^2 = {summary['s2']:.3e}, "
          f"v_bar / s^2 = {summary['v_bar_over_s2']:.3f}")
    print(f"  between-run correlation: {rho_full:.4f} (all runs), {rho_sub200:.4f} (200-run subsample)")
    print(f"  items always correct {agreement['always_correct']:.4f}, never {agreement['never_correct']:.4f}, "
          f"mixed {agreement['mixed']:.4f}")
    print("  design table n*(margin, power):")
    for d in DESIGN_MARGINS:
        print(f"    {d:>6}: " + ", ".join(f"{int(pw * 100)}% -> {design[str(d)][str(pw)]}"
                                          for pw in POWER_TARGETS))

    # --- Analysis B --------------------------------------------------------------
    print("\nAnalysis B: real seed law (disjoint arms from the seed zoo)")
    calib = calibrate_margin(s, target_n=12)
    margin_2 = calib["margin"]
    margins_b = (MARGIN_1, margin_2)
    print(f"  margin 2 calibrated to n*(80%) = 12: {margin_2} "
          f"(interval [{calib['interval'][0]:.6f}, {calib['interval'][1]:.6f}])")
    t0 = time.perf_counter()
    b_res = null_true_split(acc, N_GRID_B, margins_b, reps, rng)
    b_res["margins"] = list(margins_b)
    b_res["calibration"] = calib
    b_res["n_star_formula"] = {str(d): {str(pw): int(seeds_needed(d, pw, s, 0.0, ALPHA))
                                        for pw in POWER_TARGETS} for d in margins_b}
    b_res["first_n_reaching_80"] = {str(d): first_n_reaching(b_res["by_n"], str(d), 0.8)
                                    for d in margins_b}
    print(f"  {reps} replications per n, {time.perf_counter() - t0:.0f}s")
    for d in margins_b:
        key = str(d)
        print(f"  margin {d}: n*(80%) formula = {b_res['n_star_formula'][key]['0.8']}, "
              f"first grid n with realised rate >= 0.8: {b_res['first_n_reaching_80'][key]}")
        print(f"    {'n':>4} {'rate(D=0)':>14} {'formula t':>10} {'normal':>8} {'size +d':>14} {'size -d':>14}")
        for n in N_GRID_B:
            row = b_res["by_n"][str(n)]
            if "skipped" in row:
                continue
            r = row[key]
            print(f"    {n:>4} {fmt_rate(r['certification_rate_delta0'], r['mc_se_delta0']):>14} "
                  f"{r['formula']['t']:>10.3f} {r['formula']['normal']:>8.3f} "
                  f"{fmt_rate(r['size_at_plus_margin'], r['mc_se_plus']):>14} "
                  f"{fmt_rate(r['size_at_minus_margin'], r['mc_se_minus']):>14}")

    # --- Analysis C --------------------------------------------------------------
    print("\nAnalysis C: two-method comparisons in the hyperparameter zoo")
    hyp_rows = {t["name"]: i for i, t in enumerate(hyp_selected)}
    c_res = {}
    for pname, (a_tr, b_tr) in pair_nodes.items():
        a_idx = [hyp_rows[t["name"]] for t in a_tr]
        b_idx = [hyp_rows[t["name"]] for t in b_tr]
        acc_a = hyp_correct[a_idx].mean(axis=1)
        acc_b = hyp_correct[b_idx].mean(axis=1)
        cross = np.array([np.corrcoef(hyp_correct[i].astype(float), hyp_correct[j].astype(float))[0, 1]
                          for i in a_idx for j in b_idx])
        full = {str(d): tost_dict(acc_a, acc_b, d) for d in PAIR_MARGINS}
        s_pooled = float(np.sqrt((acc_a.var(ddof=1) + acc_b.var(ddof=1)) / 2.0))
        n_grid = list(N_GRID_C)
        n_star_full = {str(d): int(seeds_needed(d, 0.8, s_pooled, 0.0, ALPHA)) for d in PAIR_MARGINS}
        for d in PAIR_MARGINS:
            n_s = n_star_full[str(d)]
            if n_s <= min(len(acc_a), len(acc_b)) and n_s not in n_grid:
                n_grid.append(n_s)
        n_grid = sorted(n_grid)
        t0 = time.perf_counter()
        sub = two_arm_subsampling(acc_a, acc_b, n_grid, PAIR_MARGINS, reps, rng)
        pilot = pilot_then_design(acc_a, acc_b, PILOT_N0, PAIR_MARGINS, reps, rng)
        c_res[pname] = {
            "factor": PAIRS[pname][0], "level_a": PAIRS[pname][1], "level_b": PAIRS[pname][2],
            "node_a": node_id(a_tr[0]["params"]), "node_b": node_id(b_tr[0]["params"]),
            "acc_a": acc_a.tolist(), "acc_b": acc_b.tolist(),
            "seeds_a": [t["seed"] for t in a_tr], "seeds_b": [t["seed"] for t in b_tr],
            "mean_a": float(acc_a.mean()), "mean_b": float(acc_b.mean()),
            "sd_a": float(acc_a.std(ddof=1)), "sd_b": float(acc_b.std(ddof=1)),
            "s_pooled": s_pooled,
            "rho_within_a": between_run_correlation(hyp_correct[a_idx]),
            "rho_within_b": between_run_correlation(hyp_correct[b_idx]),
            "rho_between_nodes": float(cross.mean()),
            "rho_between_nodes_range": [float(cross.min()), float(cross.max())],
            "full_data_tost": full, "n_star_full_80": n_star_full,
            "subsampling": sub, "pilot_then_design": pilot,
        }
        print(f"  {pname}: mean A {acc_a.mean():.4f} (sd {acc_a.std(ddof=1):.5f}), "
              f"mean B {acc_b.mean():.4f} (sd {acc_b.std(ddof=1):.5f}), pooled s {s_pooled:.5f}; "
              f"rho between nodes {cross.mean():.3f} [{time.perf_counter() - t0:.0f}s]")
        for d in PAIR_MARGINS:
            f = full[str(d)]
            print(f"    margin {d}: delta = {f['delta']:+.4f}, 90% CI [{f['ci90'][0]:+.4f}, {f['ci90'][1]:+.4f}], "
                  f"p = {f['p_tost']:.4f}, equivalent = {f['equivalent']}, n*(80%) = {n_star_full[str(d)]}")
            rates = ", ".join(f"n={n}: {fmt_rate(sub['by_n'][str(n)][str(d)]['certification_rate'], sub['by_n'][str(n)][str(d)]['mc_se'])}"
                              for n in n_grid if "skipped" not in sub["by_n"][str(n)])
            print(f"      certification rate: {rates}")
            pl = pilot["by_margin"][str(d)]
            print(f"      pilot n0={PILOT_N0}: n* median {pl['median']:.0f}, 95% band [{pl['p2.5']:.0f}, {pl['p97.5']:.0f}]")

    # --- outputs -----------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)
    runtime = time.perf_counter() - t_start
    payload = {
        "provenance": {
            "date": today(), "root_seed": ROOT_SEED, "fast": args.fast,
            "runtime_seconds": runtime, "device": str(device),
            "software": software_versions(),
            "zenodo_record": ZENODO_RECORD, "files": provenance_files,
            "model_definition": MODEL_DEF_SOURCE,
            "archive_layout": {k: {kk: vv for kk, vv in v.items() if kk != "trial_name_map"}
                               for k, v in layouts.items()},
            "torch_load": {"weights_only": True,
                           "fallbacks_seed": seed_flags.get("weights_only_fallbacks", 0),
                           "fallbacks_hyp": hyp_flags.get("weights_only_fallbacks", 0)},
            "inference_seconds": {"seed": seed_flags["seconds"], "hyp": hyp_flags["seconds"]},
            "test_pipeline": "torchvision CIFAR10(train=False), Resize((28,28)), ToTensor, "
                             "Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)); argmax of logits",
            "m_items": M_TEST, "tolerance": TOLERANCE,
            "sign_convention": "delta = mean(level_a) - mean(level_b)",
        },
        "sanity_check": sanity,
        "analysis_A": {
            "summary": summary, "rho_all_runs": rho_full, "rho_subsample_200": rho_sub200,
            "item_agreement": agreement, "design_table": design,
            "design_margins": list(DESIGN_MARGINS), "power_targets": list(POWER_TARGETS),
            "per_run_accuracy": acc.tolist(),
        },
        "analysis_B": b_res,
        "analysis_C": c_res,
    }
    json_path = OUT_DIR / f"case2_modelzoo{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {json_path}")

    csv_path = OUT_DIR / f"case2_runs_modelzoo{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["zoo", "node", "seed", "trial", "checkpoint_index", "reported_test_acc",
                    "recomputed_accuracy"])
        for t in seed_trials:
            w.writerow(["seed", node_id(t["params"]), t["seed"], t["name"], t["checkpoint_index"],
                        f"{t['reported_test_acc']:.4f}", f"{t['recomputed_acc']:.4f}"])
        for t in hyp_selected:
            w.writerow(["hyp_fix", node_id(t["params"]), t["seed"], t["name"], t["checkpoint_index"],
                        f"{t['reported_test_acc']:.4f}", f"{t['recomputed_acc']:.4f}"])
    print(f"wrote {csv_path}")

    md = [f"# Case study 2: CIFAR-10 Model Zoos (Schuerholt et al., 2022)\n\n"]
    md.append(f"Date {today()}, root seed {ROOT_SEED}, {reps} replications, device {device}, "
              f"runtime {runtime / 60:.1f} min{' (FAST smoke run)' if args.fast else ''}. "
              f"Sign convention: delta = mean(A) - mean(B). Test set m = {M_TEST}.\n\n")
    md.append("## Data verification\n\n| Zoo | Runs | max abs diff | beyond 1e-3 | beyond 1e-4 | exact |\n|---|---:|---:|---:|---:|---:|\n")
    for k, lab in (("seed_zoo", "seed zoo"), ("hyp_zoo", "hyperparameter zoo (selected nodes)")):
        r = sanity[k]
        md.append(f"| {lab} | {r['n_runs']} | {r['max_abs_diff']:.5f} | {r['n_beyond_1e-3']} | "
                  f"{r['n_beyond_1e-4']} | {r['n_exact']} |\n")
    md.append("\n## Analysis A: seed zoo, one configuration\n\n")
    md.append(f"| R | mean | s | skewness | excess kurtosis | min | max | Shapiro p (n={summary['shapiro']['n']}) |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    md.append(f"| {summary['n_runs']} | {summary['mean']:.4f} | {s:.5f} | {summary['skewness']:+.3f} | "
              f"{summary['excess_kurtosis']:+.3f} | {summary['min']:.4f} | {summary['max']:.4f} | "
              f"{summary['shapiro']['p']:.3g} |\n\n")
    md.append("| decile | " + " | ".join(f"{q}%" for q in range(10, 100, 10)) + " |\n|---|" + "---:|" * 9 + "\n")
    md.append("| accuracy | " + " | ".join(f"{summary['deciles'][str(q)]:.4f}" for q in range(10, 100, 10)) + " |\n\n")
    md.append(f"- v_bar = {summary['v_bar']:.3e} (range of v_r [{summary['v_r_min']:.3e}, {summary['v_r_max']:.3e}]), "
              f"s^2 = {summary['s2']:.3e}, v_bar / s^2 = {summary['v_bar_over_s2']:.3f}\n")
    md.append(f"- between-run correlation of correctness: {rho_full:.4f} (all runs; 200-run subsample {rho_sub200:.4f})\n")
    md.append(f"- items always correct {agreement['always_correct']:.4f}, never correct {agreement['never_correct']:.4f}, "
              f"mixed {agreement['mixed']:.4f}\n\n")
    md.append("| margin | " + " | ".join(f"n* ({int(pw * 100)}%)" for pw in POWER_TARGETS) + " |\n|---:|" + "---:|" * len(POWER_TARGETS) + "\n")
    for d in DESIGN_MARGINS:
        md.append(f"| {d} | " + " | ".join(str(design[str(d)][str(pw)]) for pw in POWER_TARGETS) + " |\n")
    md.append("\n## Analysis B: real seed law (disjoint arms of n runs, Delta = 0 by construction)\n\n")
    md.append(f"Margins: {MARGIN_1} and {margin_2} (calibrated so that n*(80%) = 12; interval "
              f"[{calib['interval'][0]:.6f}, {calib['interval'][1]:.6f}]). Rates with MC standard errors.\n\n")
    for d in margins_b:
        key = str(d)
        md.append(f"### margin {d}: n*(80%) = {b_res['n_star_formula'][key]['0.8']}, n*(90%) = {b_res['n_star_formula'][key]['0.9']}, "
                  f"first grid n with realised rate >= 0.8: {b_res['first_n_reaching_80'][key]}\n\n")
        md.append("| n | certification rate (Delta=0) | formula (t) | formula (normal) | size at +margin | size at -margin |\n|---:|---:|---:|---:|---:|---:|\n")
        for n in N_GRID_B:
            row = b_res["by_n"][str(n)]
            if "skipped" in row:
                continue
            r = row[key]
            md.append(f"| {n} | {fmt_rate(r['certification_rate_delta0'], r['mc_se_delta0'])} | {r['formula']['t']:.3f} | "
                      f"{r['formula']['normal']:.3f} | {fmt_rate(r['size_at_plus_margin'], r['mc_se_plus'])} | "
                      f"{fmt_rate(r['size_at_minus_margin'], r['mc_se_minus'])} |\n")
        md.append("\n")
    md.append("## Analysis C: two-method comparisons (hyperparameter zoo, 10 seeds per node)\n\n")
    for pname, c in c_res.items():
        md.append(f"### {pname}: {c['node_a']} vs {c['node_b']}\n\n")
        md.append(f"- mean A {c['mean_a']:.4f} (sd {c['sd_a']:.5f}), mean B {c['mean_b']:.4f} (sd {c['sd_b']:.5f}), "
                  f"pooled s {c['s_pooled']:.5f}; correlation of correctness within A {c['rho_within_a']:.3f}, "
                  f"within B {c['rho_within_b']:.3f}, between nodes {c['rho_between_nodes']:.3f}\n\n")
        md.append("| margin | delta | 90% CI | p | equivalent | n*(80%) full s | " +
                  " | ".join(f"rate n={n}" for n in sorted(int(k) for k in c['subsampling']['by_n'])) +
                  " | pilot n* median [2.5%, 97.5%] |\n|---:|---:|---|---:|---|---:|" +
                  "---:|" * len(c['subsampling']['by_n']) + "---|\n")
        for d in PAIR_MARGINS:
            f = c["full_data_tost"][str(d)]
            rates = " | ".join(
                fmt_rate(c['subsampling']['by_n'][str(n)][str(d)]['certification_rate'],
                         c['subsampling']['by_n'][str(n)][str(d)]['mc_se'])
                if "skipped" not in c['subsampling']['by_n'][str(n)] else "-"
                for n in sorted(int(k) for k in c['subsampling']['by_n']))
            pl = c["pilot_then_design"]["by_margin"][str(d)]
            md.append(f"| {d} | {f['delta']:+.4f} | [{f['ci90'][0]:+.4f}, {f['ci90'][1]:+.4f}] | {f['p_tost']:.4f} | "
                      f"{f['equivalent']} | {c['n_star_full_80'][str(d)]} | {rates} | "
                      f"{pl['median']:.0f} [{pl['p2.5']:.0f}, {pl['p97.5']:.0f}] |\n")
        md.append("\n")
    md_path = OUT_DIR / f"case2_modelzoo{suffix}.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"total runtime {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
