#!/usr/bin/env python3
"""Plot synthetic-copying test accuracy from the released result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vo_seed_results import aggregate_series
from baseline_seed_results import aggregate_series as baseline_aggregate_series


ROOT = Path(__file__).resolve().parents[1] / "figures"
MAX_PLOT_LEN = 3000
TRAIN_MAX_LEN = 2000
REFERENCE_X_PROJECT = "bobguo__baseline_hard_alibi"
PLOT_LENGTHS = [
    1,
    97,
    201,
    305,
    401,
    497,
    601,
    697,
    801,
    897,
    993,
    1097,
    1201,
    1297,
    1401,
    1497,
    1601,
    1697,
    1801,
    1897,
    1993,
    2097,
    2201,
    2297,
    2401,
    2497,
    2601,
    2697,
    2801,
    2897,
    2993,
]

MODEL_GROUPS = [
    (
        "synthetic_standard_pe",
        [
            ("bobguo__baseline_nope", "NoPE"),
            ("bobguo__baseline_rope", "RoPE"),
            ("bobguo__baseline_alibi", "ALiBi"),
            ("bobguo__baseline_hard_alibi", "Hard-ALiBi"),
            ("bobguo__baseline_sinusoidal", "Sinusoidal"),
            ("bobguo__VO_no_scalar_", "VO-Rotation"),
        ],
        3,
    ),
]

CONDITIONS = [
    ("test-iid", "uniform random ($p_{\\mathrm{switch}}=1$)"),
    ("test-L100", "repeated ($p_{\\mathrm{switch}}\\approx0.01$)"),
    ("test-identical", "all-identical ($p_{\\mathrm{switch}}\\to 0$)"),
]

COLORS = {
    "bobguo__baseline_nope": "#4C78A8",
    "bobguo__baseline_rope": "#F58518",
    "bobguo__baseline_alibi": "#54A24B",
    "bobguo__baseline_hard_alibi": "#72B7B2",
    "bobguo__baseline_sinusoidal": "#79706E",
    "bobguo__VO_no_scalar_": "#84567A",
}

MARKERS = {
    "bobguo__baseline_nope": "o",
    "bobguo__baseline_rope": "s",
    "bobguo__baseline_alibi": "^",
    "bobguo__baseline_hard_alibi": "v",
    "bobguo__baseline_sinusoidal": "P",
    "bobguo__VO_no_scalar_": "D",
}






def filter_to_reference_grid(
    xs: list[float], ys: list[float], reference_xs: list[float]
) -> tuple[list[float], list[float]]:
    if not reference_xs:
        return xs, ys

    filtered_xs: list[float] = []
    filtered_ys: list[float] = []
    
    for ref_x in reference_xs:
        best_x = None
        best_y = None
        best_diff = 9999.0
        for x, y in zip(xs, ys):
            diff = abs(x - ref_x)
            if diff < best_diff:
                best_diff = diff
                best_x = x
                best_y = y
        if best_diff <= 4.0:
            filtered_xs.append(ref_x)
            filtered_ys.append(best_y)
            
    return filtered_xs, filtered_ys


def filter_to_plot_lengths(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    plot_grid = set(PLOT_LENGTHS)
    filtered = [
        (x, y)
        for x, y in zip(xs, ys, strict=True)
        if round(x) in plot_grid
    ]
    if not filtered:
        return [], []
    filtered_xs, filtered_ys = zip(*filtered, strict=True)
    return list(filtered_xs), list(filtered_ys)


def plot_group(
    output_stem: str,
    projects: list[tuple[str, str]],
    legend_columns: int,
    reference_xs_by_condition: dict[str, list[float]],
    show_bands: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.75), sharey=True)
    legend_handles = []
    legend_labels = []

    for ax, (condition, title) in zip(axes, CONDITIONS):
        reference_xs = reference_xs_by_condition.get(condition, [])
        for project, label in projects:
            if project == "bobguo__VO_no_scalar_":
                xs, ys, sds = aggregate_series(condition)
            else:
                xs, ys, sds = baseline_aggregate_series(project, condition)
            band_xs, sds = filter_to_reference_grid(xs, sds, reference_xs)
            band_xs, sds = filter_to_plot_lengths(band_xs, sds)
            xs, ys = filter_to_reference_grid(xs, ys, reference_xs)
            xs, ys = filter_to_plot_lengths(xs, ys)
            if not xs:
                continue
            if show_bands:
                assert xs == band_xs
                ax.fill_between(xs, [max(0, y - s) for y, s in zip(ys, sds)],
                                [min(1, y + s) for y, s in zip(ys, sds)],
                                color=COLORS[project], alpha=0.15, linewidth=0)
            (line,) = ax.plot(
                xs,
                ys,
                color=COLORS.get(project),
                marker=MARKERS.get(project, "o"),
                markevery=max(1, len(xs) // 8),
                markersize=6.5,
                linewidth=2.0,
                label=label,
                zorder=3,
            )
            if condition == CONDITIONS[0][0]:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.axvline(
            TRAIN_MAX_LEN,
            color="#333333",
            linestyle="--",
            linewidth=1.35,
            alpha=0.8,
            zorder=1,
        )
        ax.set_xlim(1, MAX_PLOT_LEN)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1000, 2000, 3000], ["1k", "2k", "3k"])
        ax.set_title(title)
        ax.set_xlabel("Test string length")
        ax.grid(True, axis="y", linewidth=0.7, color="#D7D7D7", alpha=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")

    axes[0].set_ylabel("String-level accuracy")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=legend_columns,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=2.0,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    output_path = ROOT / output_stem
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", action="store_true",
                        help="Show pointwise sample standard deviations (default: means only)")
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 17.5,
            "axes.labelsize": 18.0,
            "axes.titlesize": 17.5,
            "legend.fontsize": 17.0,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    reference_xs_by_condition = {
        condition: baseline_aggregate_series(REFERENCE_X_PROJECT, condition)[0]
        for condition, _ in CONDITIONS
    }
    for output_stem, projects, legend_columns in MODEL_GROUPS:
        plot_group(args.output_stem or output_stem, projects, legend_columns,
                   reference_xs_by_condition, show_bands=args.bands)


if __name__ == "__main__":
    main()
