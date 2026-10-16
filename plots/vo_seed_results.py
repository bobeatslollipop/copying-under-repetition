"""Load the three fixed VO checkpoints on their common, complete length grid."""

import csv
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1] / "data/synthetic"
SEEDS = (42, 43, 44)
LENGTHS = tuple(range(8, 10001, 8))
SEED42_RUNS = {"iid": "lxv7yv2h", "L100": "m24dc02m", "identical": "kiocm7l6"}


@lru_cache(maxsize=1)
def load_results():
    results = {(seed, condition): {} for seed in SEEDS for condition in SEED42_RUNS}

    def add(seed, condition, length, accuracy):
        points = results[seed, condition]
        if length in points or not 0 <= accuracy <= 1:
            raise ValueError((seed, condition, length, accuracy))
        points[length] = accuracy

    with (ROOT / "vo_seed_results.csv").open() as stream:
        for row in csv.DictReader(stream):
            add(int(row["seed"]), row["distribution"], int(row["string_length"]),
                float(row["string_accuracy"]))
    for key, points in results.items():
        if tuple(sorted(points)) != LENGTHS:
            raise ValueError(f"Incomplete evaluation grid: {key}")
    return results


def seed_series(condition, seed):
    points = load_results()[seed, condition.removeprefix("test-")]
    return list(LENGTHS), [points[n] for n in LENGTHS]


def smooth_series(xs, ys, window, boundaries=(2000, 5005)):
    """Centered moving averages, never borrowing points across regime boundaries."""
    if window < 1 or window % 2 != 1:
        raise ValueError("The smoothing window must be a positive odd integer")
    if len(xs) != len(ys):
        raise ValueError("Length and accuracy arrays must align")
    half = window // 2
    cuts = [0, *(bisect_right(xs, boundary) for boundary in boundaries), len(xs)]
    result = []
    for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
        for i in range(start, stop):
            result.append(mean(ys[max(start, i - half):min(stop, i + half + 1)]))
    return result


def aggregate_series(condition, smooth_window=1):
    curves = [seed_series(condition, seed)[1] for seed in SEEDS]
    if smooth_window != 1:
        curves = [smooth_series(LENGTHS, curve, smooth_window) for curve in curves]
    values = list(zip(*curves, strict=True))
    return list(LENGTHS), [mean(v) for v in values], [stdev(v) for v in values]
