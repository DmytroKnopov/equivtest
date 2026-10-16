"""Build the article figures at the journal's printed width.

design_curves.pdf: analytic n*(delta) design curves for the sigma_seed grid
  plus the case-study curve with its chi-square uncertainty band.
power_validation.pdf: empirical vs analytic TOST power over n, panels by true
  Delta, at theta = SIM_THETA; reads results/sim_power.json.
coverage.pdf: coverage of the two-level CI vs the naive per-image CI over the
  run count; reads results/sim_coverage_ext.json (block A).
real_seed_law.pdf: rate of declaring equivalence at true Delta = 0 and
  realised size at the boundary over the run count on real seed collections
  (the CIFAR-10 zoo at two margins, MultiBERTs SST-2); reads
  results/case2_modelzoo.json and case2_multiberts.json.

Each figure is drawn at its printed width of 119 mm (1:1 scale), lettering
9 pt (annotations 8.5 pt) in Arial (fallbacks Liberation Sans, DejaVu Sans)
embedded as TrueType, no line thinner than 0.4 pt, no figure title, panel
letters in left-aligned panel headings. Series identity is fixed across all
figures: sigma_seed -> Okabe-Ito colour, line style and marker (0.002 blue
solid circle, 0.005 vermillion dashed square, 0.01 green dotted triangle),
so the encoding survives printing in black and white; the real-seed-law
figure compares three collections instead and carries its own triple (orange
solid diamond, reddish-purple dashed pentagon, sky-blue dotted inverted
triangle). Empirical estimates are open markers with MC error bars, analytic
curves are lines. The captions in the article are the single source of
description. A 150 dpi PNG preview is written next to each PDF.

Run from repo root (no argument builds every figure; naming builders,
e.g. ``fig4``, rebuilds only those; fig1 .. fig4 follow the figure numbers
of the article):
    python experiments/build_figures.py [fig1 ... fig4]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
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
SIM_THETA = 0.75

# Printed figure width of the journal (119 mm).
FIG_WIDTH = 4.685
PREVIEW_DPI = 150

# Fixed series identity for sigma_seed (never re-assigned between figures):
# Okabe-Ito colour on top of line style and marker.
SIGMA_STYLE = {
    0.002: {"color": "#0072B2", "ls": "-", "marker": "o"},
    0.005: {"color": "#D55E00", "ls": "--", "marker": "s"},
    0.01: {"color": "#009E73", "ls": ":", "marker": "^"},
}
# Multiplicative horizontal offset of the empirical markers (the run-count
# axes are logarithmic) so that the three series' error bars at the same n
# do not sit on top of each other.
SIGMA_DODGE = {0.002: 0.955, 0.005: 1.0, 0.01: 1.047}
# The real-seed-law figure compares three collections rather than the
# sigma_seed grid, so it carries its own encoding (Okabe-Ito colours on top
# of line style and marker): zoo at delta = 0.01, zoo at delta = 0.0173,
# MultiBERTs SST-2.
CASE2_STYLE = [
    {"color": "#E69F00", "ls": "-", "marker": "D"},
    {"color": "#CC79A7", "ls": "--", "marker": "p"},
    {"color": "#56B4E9", "ls": ":", "marker": "v"},
]

REF_COLOR = "0.55"     # reference lines (alpha, target power, nominal level)
NOTE_COLOR = "0.4"     # in-panel notes; darker than 0.4 keeps contrast >= 4.5:1
NOTE_SIZE = 8.5
CURVE_LW = 1.4
MARKER_SIZE = 5.0
ERRORBAR_KW = {"markerfacecolor": "white", "markersize": MARKER_SIZE,
               "capsize": 2.2, "linewidth": 1.0, "linestyle": "none"}


def _first_available(families: list[str]) -> str:
    """First installed family of the list (the journal asks for Arial)."""
    for family in families:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


SANS = _first_available(["Arial", "Liberation Sans", "DejaVu Sans"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": SANS,
    "mathtext.it": SANS + ":italic",
    "mathtext.bf": SANS + ":bold",
    "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "legend.framealpha": 1.0,
})


def _sigma_label(s: float) -> str:
    return rf"$\sigma_{{\mathrm{{seed}}}} = {s}$"


def _save(fig, name: str) -> Path:
    """PDF at exactly the figure size, PNG preview next to it."""
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=PREVIEW_DPI)
    plt.close(fig)
    return out


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


def fig1_seeds_needed() -> None:
    power_target = 0.8
    v_bar = None
    sim = TABLES / "sim_power.json"
    if sim.exists():
        v_bar = json.loads(sim.read_text(encoding="utf-8"))["config"][
            "v_bar_per_theta"].get(str(SIM_THETA))
    if v_bar is None:
        v_bar = 1.16e-05  # theta=0.75, m0=m1=10^4 (sim_power estimate)

    deltas = np.linspace(0.004, 0.03, 60)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.0), layout="constrained")

    # 95% chi^2 band for the case-study n*: s^2 rests on ~1.2 df at n = 2,
    # and n* is proportional to s^2, so the band is a constant-width corridor
    # on the log axis.
    s2_lo, s2_hi = _case_study_s2_band()
    band_lo = [_seeds_needed_fast(float(d), power_target, s2_lo, ALPHA)
               for d in deltas]
    band_hi = [_seeds_needed_fast(float(d), power_target, s2_hi, ALPHA)
               for d in deltas]
    # In Arial the text asterisk already sits at superscript height, so n*
    # is set as an italic n followed by a text asterisk (a mathtext
    # superscript would float above the line at reduced size).
    ax.fill_between(deltas, band_lo, band_hi, color="0.9", zorder=0,
                    label=r"case study: 95% band for $n$*")

    for sigma, style in SIGMA_STYLE.items():
        ns = [seeds_needed(float(d), power_target, sigma, v_bar, ALPHA)
              for d in deltas]
        ax.plot(deltas, ns, style["ls"], color=style["color"],
                linewidth=CURVE_LW, label=_sigma_label(sigma),
                marker=style["marker"], markersize=4,
                markerfacecolor="white", markevery=7)

    # Case study: pooled seed-level SD of macro-5, four runs.
    sd1 = _case_study_pooled_sd()
    ns1 = [seeds_needed(float(d), power_target, sd1, 0.0, ALPHA) for d in deltas]
    ax.plot(deltas, ns1, "-", color="0.0", linewidth=2.2,
            label=rf"case study: $s = {sd1:.4f}$")
    n_star = seeds_needed(MARGIN, power_target, sd1, 0.0, ALPHA)
    ax.plot([MARGIN], [n_star], marker="*", color="0.0", markersize=11,
            linestyle="none", zorder=5)
    n_lo = _seeds_needed_fast(MARGIN, power_target, s2_lo, ALPHA)
    n_hi = _seeds_needed_fast(MARGIN, power_target, s2_hi, ALPHA)
    ax.annotate(
        rf"$\delta = {MARGIN}$: $n$*$= {n_star}$,"
        rf" band $[{n_lo},\ {n_hi}]$"
        "\n"
        rf"(shifted-$t$ power at $n = 2$: $\approx 0$)",
        xy=(MARGIN, n_star), xytext=(0.0125, n_star * 3),
        fontsize=NOTE_SIZE, arrowprops={"arrowstyle": "->", "color": "0.3",
                                        "linewidth": 0.8},
    )
    ax.axvline(MARGIN, color="0.75", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_yscale("log")
    # Ticks up to the top of the band (about 22 000 at delta = 0.004); a tick
    # beyond the autoscaled range would stretch the axis, so it is dropped.
    top = ax.get_ylim()[1]
    ticks = [t for t in (2, 5, 10, 20, 50, 100, 300, 1000, 3000, 10000, 30000)
             if t <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    # Below five runs the formula is not relied on (Remark on validity):
    # hatch that band and name it in the legend (a label inside the band
    # would collide with the curves).
    bottom = ax.get_ylim()[0]
    ax.axhspan(bottom, 5, facecolor="none", edgecolor="0.6", hatch="////",
               linewidth=0, zorder=0,
               label="below five runs: formula not relied on")
    ax.set_ylim(bottom, top)
    ax.set_xlabel(r"equivalence margin $\delta$")
    ax.set_ylabel(rf"runs per method for power {power_target}")
    ax.legend(loc="upper right", frameon=True)
    out = _save(fig, "design_curves")
    print(f"wrote {out} (n* at delta={MARGIN}: {n_star}, band [{n_lo}, {n_hi}])")


def fig2_power_validation() -> None:
    path = TABLES / "sim_power.json"
    if not path.exists():
        print("power_validation: results/sim_power.json not found -- skipped")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c["theta"] == SIM_THETA]
    # Panels (a)-(b): power at Delta = 0 and 0.005 (full scale, three series
    # per panel). Panels (c)-(e): realised size at the boundary Delta = +delta
    # on an enlarged scale, one panel per sigma -- at the boundary all three
    # series converge to alpha, so drawn together they would sit on top of
    # each other. The Delta = -delta and Delta = 0.02 cells stay in the text:
    # the -delta panel would mirror the +delta one, and beyond the margin
    # every curve is flat at zero.
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
                    linewidth=CURVE_LW, label=_sigma_label(sigma))
            ax.errorbar(
                ns * (SIGMA_DODGE[sigma] if dodge else 1.0),
                [c["power_empirical"] for c in pts],
                yerr=[3 * c["power_mc_se"] for c in pts],
                fmt=style["marker"], color=style["color"], **ERRORBAR_KW,
            )
        ax.axhline(ALPHA, color=REF_COLOR, linestyle="-.", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(n_ticks)
        ax.set_xticklabels([str(n) for n in n_ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1.75, 23.0)

    fig = plt.figure(figsize=(FIG_WIDTH, 4.1), layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.62])
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6], sharey=ax_a)
    for ax, letter, delta in zip((ax_a, ax_b), "ab", (0.0, 0.005)):
        draw(ax, delta, list(SIGMA_STYLE), dodge=True)
        ax.set_title(rf"({letter}) true $\Delta = {delta:g}$")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"runs per method $n$")
    ax_b.tick_params(labelleft=False)
    size_axes = []
    for k, (letter, sigma) in enumerate(zip("cde", SIGMA_STYLE)):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2],
                             sharey=size_axes[0] if size_axes else None)
        draw(ax, MARGIN, [sigma], dodge=False)
        ax.set_title(rf"({letter}) " + _sigma_label(sigma))
        size_axes.append(ax)
    size_axes[0].set_ylim(-0.005, 0.105)
    size_axes[0].set_yticks([0.0, 0.05, 0.10])
    for ax in size_axes[1:]:
        ax.tick_params(labelleft=False)
    size_axes[1].set_xlabel(r"runs per method $n$")
    size_axes[-1].text(n_ticks[-1], ALPHA + 0.022, rf"$\alpha = {ALPHA}$",
                       color=NOTE_COLOR, fontsize=NOTE_SIZE, ha="right",
                       va="bottom")
    fig.supylabel("probability of declaring equivalence", fontsize=9)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=True)
    out = _save(fig, "power_validation")
    print(f"wrote {out}")


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
    # shared axis would squeeze panel (a) into its top tenth.
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.5), sharey=False,
                             layout="constrained")
    panels = [
        ("coverage_two_level", "coverage_mc_se_two",
         r"(a) two-level (seed $\times$ image)"),
        ("coverage_naive", "coverage_mc_se_naive", "(b) naive per-image"),
    ]
    for ax, (key, se_key, title) in zip(axes, panels):
        for sigma, style in SIGMA_STYLE.items():
            pts = sorted((c for c in cells if c["sigma_seed"] == sigma),
                         key=lambda c: c["n"])
            ax.errorbar(
                [c["n"] * SIGMA_DODGE[sigma] for c in pts], [c[key] for c in pts],
                yerr=[3 * c[se_key] for c in pts],
                marker=style["marker"], color=style["color"],
                linestyle=style["ls"], linewidth=1.2,
                markerfacecolor="white", markersize=MARKER_SIZE, capsize=2.2,
                label=_sigma_label(sigma),
            )
        ax.axhline(0.95, color=REF_COLOR, linestyle="--", linewidth=0.9)
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
    # Zoom indicator: the whole panel (a) lives inside this band of panel (b).
    lo, hi = axes[0].get_ylim()
    axes[1].axhspan(lo, hi, color="0.92", zorder=0, linewidth=0)
    axes[1].text(1.85, hi - 0.006, "range of (a)", color=NOTE_COLOR,
                 fontsize=NOTE_SIZE, va="top", ha="left")
    axes[1].text(13.3, 0.95 + 0.006, "nominal 0.95", color=NOTE_COLOR,
                 fontsize=NOTE_SIZE, va="bottom", ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=True)
    out = _save(fig, "coverage")
    print(f"wrote {out}")


def fig4_real_seed_law() -> None:
    """Real seed collections: rate of declaring equivalence at true Delta = 0
    (markers: resampled disjoint arms with +/-3 MC-SE bars; lines: the
    shifted-t formula through the grid) and the realised size at
    Delta = +/-delta (mean of the two boundary rates) over the run count;
    reads results/case2_modelzoo.json (analysis B) and
    case2_multiberts.json (SST-2 null-true check)."""
    zoo_path = TABLES / "case2_modelzoo.json"
    bert_path = TABLES / "case2_multiberts.json"
    if not (zoo_path.exists() and bert_path.exists()):
        print("real_seed_law: results/case2_modelzoo.json or "
              "case2_multiberts.json not found -- skipped")
        return
    zoo = json.loads(zoo_path.read_text(encoding="utf-8"))["analysis_B"]["by_n"]
    sst2 = json.loads(bert_path.read_text(encoding="utf-8"))[
        "tasks"]["SST-2"]["null_true_check"]["by_n"]
    series = [
        (zoo, "0.01", r"zoo, $\delta = 0.01$"),
        (zoo, "0.0173", r"zoo, $\delta = 0.0173$"),
        (sst2, "0.01", r"SST-2, $\delta = 0.01$"),
    ]
    n_ticks = sorted({int(n) for cells, _, _ in series for n in cells})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.5),
                                     layout="constrained")
    handles, labels = [], []
    for (cells, margin, label), style, dodge in zip(
            series, CASE2_STYLE, SIGMA_DODGE.values()):
        ns = np.array(sorted(int(n) for n in cells))
        rows = [cells[str(n)][margin] for n in ns]
        marker_kw = dict(fmt=style["marker"], color=style["color"], **ERRORBAR_KW)
        ax_a.plot(ns, [r["formula"]["t"] for r in rows], style["ls"],
                  color=style["color"], linewidth=CURVE_LW)
        ax_a.errorbar(ns * dodge, [r["certification_rate_delta0"] for r in rows],
                      yerr=[3 * r["mc_se_delta0"] for r in rows], **marker_kw)
        ax_b.errorbar(
            ns * dodge,
            [(r["size_at_plus_margin"] + r["size_at_minus_margin"]) / 2
             for r in rows],
            yerr=[3 * 0.5 * np.hypot(r["mc_se_plus"], r["mc_se_minus"])
                  for r in rows],
            **marker_kw)
        handles.append(Line2D([], [], color=style["color"], linestyle=style["ls"],
                              linewidth=CURVE_LW, marker=style["marker"],
                              markerfacecolor="white", markersize=MARKER_SIZE))
        labels.append(label)
    ax_a.axhline(0.8, color=REF_COLOR, linestyle="-.", linewidth=0.8)
    ax_b.axhline(ALPHA, color=REF_COLOR, linestyle="-.", linewidth=0.8)
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xticks(n_ticks)
        ax.set_xticklabels([str(n) for n in n_ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1.75, 57.5)
        ax.set_xlabel(r"runs per method $n$")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_ylabel("probability of declaring equivalence")
    ax_a.set_title(r"(a) true $\Delta = 0$")
    ax_b.set_ylim(0.0, 0.10)
    ax_b.set_yticks([0.0, 0.05, 0.10])
    ax_b.set_ylabel("realised size at the boundary")
    ax_b.set_title(r"(b) $\Delta = \pm\delta$")
    ax_b.text(55.0, ALPHA + 0.011, rf"$\alpha = {ALPHA}$", color=NOTE_COLOR,
              fontsize=NOTE_SIZE, ha="right", va="bottom")
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=True)
    out = _save(fig, "real_seed_law")
    print(f"wrote {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    builders = {
        "fig1": fig1_seeds_needed,
        "fig2": fig2_power_validation,
        "fig3": fig3_coverage,
        "fig4": fig4_real_seed_law,
    }
    unknown = set(sys.argv[1:]) - set(builders)
    if unknown:
        sys.exit(f"unknown figure builder(s): {', '.join(sorted(unknown))}; "
                 f"choose from {', '.join(builders)}")
    for name in sys.argv[1:] or list(builders):
        builders[name]()
