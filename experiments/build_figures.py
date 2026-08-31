"""Build the article figures (black-and-white, as printed).

design_curves.pdf: analytic n*(delta) design curves for the sigma_seed grid
  plus the case-study curve with its chi-square uncertainty band.
power_validation.pdf: empirical vs analytic TOST power over n, panels by true
  Delta, at theta = FIG1_THETA; reads results/sim_power.json.
coverage.pdf: coverage of the two-level CI vs the naive per-image CI over the
  run count; reads results/sim_coverage_ext.json (block A).

Series identity is fixed across all figures (sigma_seed -> gray level,
line style, marker); empirical estimates are open markers with MC error
bars, analytic curves are lines. Figures carry no internal titles -- the
captions in main.tex are the single source of description.

Run from repo root:
    python experiments/build_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equivtest.power import seeds_needed, tost_power  # noqa: E402

TABLES = ROOT / "results"
FIG_DIR = ROOT / "figures"
CASE_JSON = TABLES / "case_study.json"

MARGIN = 0.01
ALPHA = 0.05
FIG1_THETA = 0.75

# Fixed series identity for sigma_seed (never re-assigned between figures).
SIGMA_STYLE = {
    0.002: {"color": "0.3", "ls": "-", "marker": "o"},
    0.005: {"color": "0.42", "ls": "--", "marker": "s"},
    0.01: {"color": "0.62", "ls": ":", "marker": "^"},
}
# Multiplicative horizontal offset of the empirical markers (the run-count
# axes are logarithmic) so that the three series' error bars at the same n
# do not sit on top of each other.
SIGMA_DODGE = {0.002: 0.955, 0.005: 1.0, 0.01: 1.047}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
})


def _sigma_label(s: float) -> str:
    return rf"$\sigma_{{\mathrm{{seed}}}} = {s}$"


def fig1_power_validation() -> None:
    path = TABLES / "sim_power.json"
    if not path.exists():
        print("power_validation: results/sim_power.json not found -- skipped")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c["theta"] == FIG1_THETA]
    # Top row: power at Delta = 0 and 0.005 (full scale, three series per
    # panel). Bottom row: realised size at the boundary Delta = +delta on an
    # enlarged scale, one panel per sigma -- at the boundary all three series
    # converge to alpha, so drawn together they would sit on top of each
    # other. The Delta = -delta and Delta = 0.02 cells stay in the text: the
    # -delta panel would mirror the +delta one, and beyond the margin every
    # curve is flat at zero.
    n_ticks = sorted({c["n"] for c in cells})

    def draw(ax, delta, sigmas, dodge):
        sub = [c for c in cells if c["delta"] == delta]
        for sigma in sigmas:
            style = SIGMA_STYLE[sigma]
            pts = sorted((c for c in sub if c["sigma_seed"] == sigma),
                         key=lambda c: c["n"])
            if not pts:
                continue
            ns = np.array([c["n"] for c in pts])
            n_dense = np.arange(2, ns.max() + 1)
            analytic = [
                tost_power(delta, MARGIN, int(n), sigma, pts[0]["v_bar"], ALPHA)
                for n in n_dense
            ]
            ax.plot(n_dense, analytic, style["ls"], color=style["color"],
                    linewidth=1.6, label=_sigma_label(sigma))
            ax.errorbar(
                ns * (SIGMA_DODGE[sigma] if dodge else 1.0),
                [c["power_empirical"] for c in pts],
                yerr=[3 * c["power_mc_se"] for c in pts],
                fmt=style["marker"], color=style["color"],
                markerfacecolor="white", markersize=5.5,
                capsize=2.5, linewidth=1.1, linestyle="none",
            )
        ax.axhline(ALPHA, color="0.55", linestyle="-.", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(n_ticks)
        ax.set_xticklabels([str(n) for n in n_ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1.75, 23.0)

    fig = plt.figure(figsize=(7.0, 4.3), dpi=150)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.62])
    ax_left = fig.add_subplot(gs[0, 0:3])
    ax_right = fig.add_subplot(gs[0, 3:6], sharey=ax_left)
    for ax, delta in zip((ax_left, ax_right), (0.0, 0.005)):
        draw(ax, delta, list(SIGMA_STYLE), dodge=True)
        ax.set_title(rf"true $\Delta = {delta}$")
        ax.set_ylim(-0.03, 1.03)
    ax_right.tick_params(labelleft=False)
    size_axes = []
    for k, sigma in enumerate(SIGMA_STYLE):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2],
                             sharey=size_axes[0] if size_axes else None)
        draw(ax, MARGIN, [sigma], dodge=False)
        ax.set_title(r"$\Delta = \delta$, " + _sigma_label(sigma))
        size_axes.append(ax)
    size_axes[0].set_ylim(-0.005, 0.105)
    size_axes[0].set_yticks([0.0, 0.05, 0.10])
    for ax in size_axes[1:]:
        ax.tick_params(labelleft=False)
    size_axes[1].set_xlabel(r"runs per method $n$")
    size_axes[-1].text(n_ticks[-1], ALPHA + 0.022, rf"$\alpha = {ALPHA}$",
                       color="0.4", fontsize=8, ha="right", va="bottom")
    fig.supylabel("probability of declaring equivalence", fontsize=9.5)
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    out = FIG_DIR / "power_validation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _case_study_pooled_sd() -> float:
    case = json.loads(CASE_JSON.read_text(encoding="utf-8"))
    return float(case["seed_level"]["s_pooled"])


def _case_study_s2_band() -> tuple[float, float]:
    """95% chi^2 interval for the case-study s^2 (df from the Welch-TOST)."""
    case = json.loads(CASE_JSON.read_text(encoding="utf-8"))
    s2 = float(case["seed_level"]["s2_pooled"])
    df = float(case["tost"]["df"])
    return df * s2 / chi2.ppf(0.975, df), df * s2 / chi2.ppf(0.025, df)


def _seeds_needed_fast(margin: float, power: float, s2: float, alpha: float) -> int:
    """seeds_needed by bisection (power is monotone in n), for large n*."""
    sigma = float(np.sqrt(s2))
    lo, hi = 2, 2
    while tost_power(0.0, margin, hi, sigma, 0.0, alpha) < power:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if tost_power(0.0, margin, mid, sigma, 0.0, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fig2_seeds_needed() -> None:
    power_target = 0.8
    v_bar = None
    sim = TABLES / "sim_power.json"
    if sim.exists():
        v_bar = json.loads(sim.read_text(encoding="utf-8"))["config"][
            "v_bar_per_theta"].get(str(FIG1_THETA))
    if v_bar is None:
        v_bar = 1.16e-05  # theta=0.75, m0=m1=10^4 (sim_power estimate)

    deltas = np.linspace(0.004, 0.03, 60)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)

    # 95% chi^2 band for the case-study n*: s^2 rests on ~1.2 df at n = 2,
    # and n* is proportional to s^2, so the band is a constant-width corridor
    # on the log axis.
    s2_lo, s2_hi = _case_study_s2_band()
    band_lo = [_seeds_needed_fast(float(d), power_target, s2_lo, ALPHA)
               for d in deltas]
    band_hi = [_seeds_needed_fast(float(d), power_target, s2_hi, ALPHA)
               for d in deltas]
    ax.fill_between(deltas, band_lo, band_hi, color="0.9", zorder=0,
                    label=r"case study: 95% band for $n^*$")

    for sigma, style in SIGMA_STYLE.items():
        ns = [seeds_needed(float(d), power_target, sigma, v_bar, ALPHA)
              for d in deltas]
        ax.plot(deltas, ns, style["ls"], color=style["color"],
                linewidth=1.6, label=_sigma_label(sigma),
                marker=style["marker"], markersize=4,
                markerfacecolor="white", markevery=7)

    # Case study: pooled seed-level SD of macro-5, four runs.
    sd1 = _case_study_pooled_sd()
    ns1 = [seeds_needed(float(d), power_target, sd1, 0.0, ALPHA) for d in deltas]
    ax.plot(deltas, ns1, "-", color="0.0", linewidth=2.4,
            label=rf"case study: $s = {sd1:.4f}$")
    n_star = seeds_needed(MARGIN, power_target, sd1, 0.0, ALPHA)
    ax.plot([MARGIN], [n_star], marker="*", color="0.0", markersize=13,
            linestyle="none", zorder=5)
    n_lo = _seeds_needed_fast(MARGIN, power_target, s2_lo, ALPHA)
    n_hi = _seeds_needed_fast(MARGIN, power_target, s2_hi, ALPHA)
    ax.annotate(
        rf"$\delta = {MARGIN}$: $n^* = {n_star}$,"
        rf" band $[{n_lo},\ {n_hi}]$"
        "\n"
        rf"(at $n = 2$: power $\approx 0$)",
        xy=(MARGIN, n_star), xytext=(0.0135, n_star * 6),
        fontsize=8.5, arrowprops={"arrowstyle": "->", "color": "0.3",
                                  "linewidth": 0.8},
    )
    ax.axvline(MARGIN, color="0.75", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100, 300, 1000, 3000])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "300", "1000", "3000"])
    ax.set_xlabel(r"equivalence margin $\delta$")
    ax.set_ylabel(rf"runs per method for power {power_target}")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "design_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (n* at delta={MARGIN}: {n_star}, band [{n_lo}, {n_hi}])")


def fig3_coverage() -> None:
    path = TABLES / "sim_coverage_ext.json"
    if not path.exists():
        print("coverage: results/sim_coverage_ext.json not found -- skipped")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c["block"] == "A"]
    n_values = sorted({c["n"] for c in cells})

    # The two panels live on different vertical scales: the two-level interval
    # stays within 0.8-1.0 while the naive one collapses towards 0.5, so a
    # shared axis would squeeze the left panel into its top tenth.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=150, sharey=False)
    panels = [
        ("coverage_two_level", "coverage_mc_se_two", "two-level (seed × image)"),
        ("coverage_naive", "coverage_mc_se_naive", "naive per-image"),
    ]
    for ax, (key, se_key, title) in zip(axes, panels):
        for sigma, style in SIGMA_STYLE.items():
            pts = sorted((c for c in cells if c["sigma_seed"] == sigma),
                         key=lambda c: c["n"])
            ax.errorbar(
                [c["n"] * SIGMA_DODGE[sigma] for c in pts], [c[key] for c in pts],
                yerr=[3 * c[se_key] for c in pts],
                marker=style["marker"], color=style["color"],
                linestyle=style["ls"], linewidth=1.3,
                markerfacecolor="white", markersize=5.5, capsize=2.5,
                label=_sigma_label(sigma),
            )
        ax.axhline(0.95, color="0.55", linestyle="--", linewidth=0.9,
                   label="nominal 0.95")
        ax.set_xscale("log")
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1.75, 13.8)
        ax.set_xlabel(r"runs per method $n$")
        ax.set_ylabel(r"coverage of true $\Delta$")
        ax.set_title(title)
    axes[0].set_ylim(0.79, 1.01)
    axes[0].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    axes[1].set_ylim(0.45, 1.01)
    axes[1].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Zoom indicator: the whole left panel lives inside this band of the right.
    lo, hi = axes[0].get_ylim()
    axes[1].axhspan(lo, hi, color="0.92", zorder=0, linewidth=0)
    axes[1].text(1.85, hi - 0.006, "range of the left panel", color="0.35",
                 fontsize=7.5, va="top", ha="left")
    axes[0].legend(loc="lower right", frameon=True)
    fig.tight_layout()
    out = FIG_DIR / "coverage.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    fig1_power_validation()
    fig2_seeds_needed()
    fig3_coverage()
