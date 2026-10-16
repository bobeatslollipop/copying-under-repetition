#!/usr/bin/env python3
"""Plot the recorded seed-42 VO training loss."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator


ROOT = Path(__file__).resolve().parents[1] / "figures"
DATA_ROOT = Path(__file__).resolve().parents[1] / "data/synthetic"
OUTPUT_STEM = ROOT / "synthetic_training_loss"
METRIC = "train_loss"
MAX_UPDATE = 200_000

PROJECTS = [
    ("bobguo__VO_no_scalar_", "VO-Rotation", "#B279A2", "D"),
]




def read_series(project: str) -> tuple[list[float], list[float]]:
    by_step: dict[float, tuple[str, float]] = {}
    for path, created_at in [(DATA_ROOT / "vo_training_seed42.csv", "")]:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "optimizer_update" not in fieldnames or METRIC not in fieldnames:
                continue
            for row in reader:
                try:
                    step = float(row["optimizer_update"])
                    value = float(row[METRIC])
                except (KeyError, TypeError, ValueError):
                    continue
                if step <= 0 or step > MAX_UPDATE or value <= 0:
                    continue
                old = by_step.get(step)
                if old is None or created_at >= old[0]:
                    by_step[step] = (created_at, value)

    ordered = sorted(by_step)
    return ordered, [by_step[step][1] for step in ordered]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 14.5,
            "axes.labelsize": 14.5,
            "axes.titlesize": 14.5,
            "legend.fontsize": 13.2,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    for project, label, color, marker in PROJECTS:
        xs, ys = read_series(project)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.0,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlim(1_000, MAX_UPDATE)
    ax.set_xticks([1_000, 10_000, 100_000])
    ax.set_xticklabels(["1k", "10k", "100k"])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Training loss")
    ax.grid(True, axis="y", which="major", linewidth=0.7, color="#D7D7D7", alpha=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    fig.tight_layout()
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
