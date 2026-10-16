#!/usr/bin/env python3
"""Plot the main VO-Rotation length-generalization figure and per-seed appendix."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from vo_seed_results import SEEDS, aggregate_series, seed_series

ROOT = Path(__file__).resolve().parents[1] / "figures"
MAX_PLOT_LEN = 10000
TRAIN_MAX_LEN = 2000
PERIOD_PRODUCT = 5005
SMOOTH_WINDOW = 1

CONDITIONS = [
    ("test-iid", "uniform random ($p_{\\mathrm{switch}}=1$)"),
    ("test-L100", "repeated ($p_{\\mathrm{switch}}\\approx0.01$)"),
    ("test-identical", "all-identical ($p_{\\mathrm{switch}}\\to 0$)"),
]

def style_axis(ax):
    ax.axvline(TRAIN_MAX_LEN, color="#333333", linestyle="--", linewidth=1.35, zorder=1)
    ax.axvline(PERIOD_PRODUCT, color="#C74B45", linestyle="-.", linewidth=1.35, zorder=1)
    ax.set_xlim(1, MAX_PLOT_LEN)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([2500, 5000, 7500, 10000], ["2.5k", "5k", "7.5k", "10k"])
    ax.grid(True, axis="y", linewidth=0.7, color="#D7D7D7", alpha=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")


def reference_legend(fig):
    handles = [Line2D([], [], color="#333333", linestyle="--", linewidth=1.35),
               Line2D([], [], color="#C74B45", linestyle="-.", linewidth=1.35)]
    fig.legend(handles, ["Training limit (2000)", "Period product (5005)"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.5, columnspacing=2.0)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
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

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 4.15), sharey=True)

    for ax, (condition, title) in zip(axes, CONDITIONS):
        xs, ys, sds = aggregate_series(condition, smooth_window=SMOOTH_WINDOW)
        ax.fill_between(xs, [max(0, y - s) for y, s in zip(ys, sds)],
                        [min(1, y + s) for y, s in zip(ys, sds)],
                        color="#B279A2", alpha=0.22, linewidth=0, zorder=2)
        ax.plot(
            xs,
            ys,
            color="#84567A",
            linewidth=1.0,
            zorder=3,
        )
        style_axis(ax)
        ax.set_title(title)
        ax.set_xlabel("Test string length")

    axes[0].set_ylabel("String-level accuracy")
    
    reference_legend(fig)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    output_path = ROOT / "synthetic_vo_generalization_10000"
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("Saved generalization plot to synthetic_vo_generalization_10000.pdf/png")

    fig, axes = plt.subplots(3, 3, figsize=(10.7, 6.8), sharex=True, sharey=True)
    for row, seed in enumerate(SEEDS):
        for col, (condition, title) in enumerate(CONDITIONS):
            ax = axes[row, col]
            xs, ys = seed_series(condition, seed)
            ax.plot(xs, ys, color="#84567A", linewidth=0.7, zorder=3)
            style_axis(ax)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"Seed {seed}", rotation=0, labelpad=38, va="center")
    fig.supylabel("String-level accuracy", x=0.015, fontsize=18.0)
    fig.supxlabel("Test string length", y=0.075, fontsize=18.0)
    reference_legend(fig)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.93, bottom=0.19,
                        hspace=0.28, wspace=0.18)
    output_path = ROOT / "synthetic_vo_generalization_per_seed"
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
