#!/usr/bin/env python3
"""Generate the main-text LLM copying figure from CSV results."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LENGTHS = [30, 100, 300, 1000, 3000]
CONDITIONS = ["iid", "geometric_L=100.0", "geometric_L=inf"]
CONDITION_LABELS = {
    "iid": "uniform random ($p_{\\mathrm{switch}}=1$)",
    "geometric_L=100.0": "repeated ($p_{\\mathrm{switch}}\\approx0.01$)",
    "geometric_L=inf": "all-identical ($p_{\\mathrm{switch}}\\to 0$)",
}
VARIANT_LABELS = {
    "397b": "Qwen3.5\n(no thinking)",
    "deepseek-v4-pro-no-thinking": "DeepSeek V4 Pro\n(no thinking)",
    "gemini-3.1-pro-low-thinking": "Gemini 3.1 Pro\n(low thinking)",
    "gemini-3.1-pro-medium-thinking": "Gemini 3.1 Pro\n(medium thinking)",
    "gpt-5.5": "GPT-5.5\n(no reasoning)",
    "gpt-5.5-medium-reasoning": "GPT-5.5\n(medium reasoning)",
}
VARIANT_ORDER = [
    "397b",
    "deepseek-v4-pro-no-thinking",
    "gemini-3.1-pro-low-thinking",
    "gemini-3.1-pro-medium-thinking",
    "gpt-5.5",
    "gpt-5.5-medium-reasoning",
]
COLORS = {
    "397b": "#4E79A7",
    "deepseek-v4-pro-no-thinking": "#F28E2B",
    "gemini-3.1-pro-low-thinking": "#59A14F",
    "gemini-3.1-pro-medium-thinking": "#B6992D",
    "gpt-5.5": "#E15759",
    "gpt-5.5-medium-reasoning": "#AF7AA1",
}
MARKERS = {
    "397b": "o",
    "deepseek-v4-pro-no-thinking": "s",
    "gemini-3.1-pro-low-thinking": "^",
    "gemini-3.1-pro-medium-thinking": "v",
    "gpt-5.5": "D",
    "gpt-5.5-medium-reasoning": "P",
}


def read_results(data_dir: Path) -> dict[tuple[str, str, int], float]:
    """Return mean string accuracy keyed by (variant, condition, length)."""
    csv.field_size_limit(sys.maxsize)
    sums: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for path in sorted(data_dir.glob("*.csv")):
        variant = path.stem
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                condition = row.get("condition", "")
                if condition not in CONDITIONS:
                    continue
                try:
                    length = int(float(row["string_length"]))
                    str_acc = float(row["str_acc"])
                except (KeyError, TypeError, ValueError):
                    continue
                if length in LENGTHS:
                    sums[(variant, condition, length)].append(str_acc)
    return {key: sum(vals) / len(vals) for key, vals in sums.items() if vals}


def present_variants(means: dict[tuple[str, str, int], float]) -> list[str]:
    variants = {variant for variant, _, _ in means}
    ordered = [v for v in VARIANT_ORDER if v in variants]
    ordered.extend(sorted(variants - set(ordered)))
    return ordered


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/llm"
    output_dir = root / "figures"
    output_dir.mkdir(exist_ok=True)
    means = read_results(data_dir)
    variants = present_variants(means)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 18.0,
            "axes.titlesize": 17.5,
            "axes.labelsize": 18.0,
            "legend.fontsize": 16.0,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.0), sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        ax.set_facecolor("white")
        for variant in variants:
            ys = [means.get((variant, condition, length), float("nan")) for length in LENGTHS]
            ax.plot(
                LENGTHS,
                ys,
                marker=MARKERS.get(variant, "o"),
                color=COLORS.get(variant),
                linewidth=2.2,
                markersize=7.0,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=VARIANT_LABELS.get(variant, variant),
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xticks(LENGTHS)
        ax.set_xticklabels(["30", "100", "300", "1k", "3k"])
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xlabel("String length")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.7, color="#D7D7D7", alpha=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#4A4A4A")
            ax.spines[side].set_linewidth(1.1)
        ax.tick_params(axis="both", width=1.1, length=5.2)
    axes[0].set_ylabel("String-level accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=4.0,
        handlelength=1.8,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.43, wspace=0.13)
    fig.savefig(output_dir / "accuracy_plot_main_all_conditions.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
