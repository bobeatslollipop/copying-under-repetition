"""Load validated baseline runs on the common three-seed evaluation grid."""
import csv
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1] / "data/synthetic"
SEEDS = (42, 43, 44)
LENGTHS = tuple(range(1, 3000, 8))
MODELS = ("nope", "rope", "alibi", "hard_alibi", "sinusoidal")
CONDITIONS = ("iid", "L100", "identical")


@lru_cache(maxsize=1)
def load_results():
    results = {(model, seed, condition): {} for model in MODELS
               for seed in SEEDS for condition in CONDITIONS}
    with (ROOT / "baseline_seed_results.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = row["model"], int(row["seed"]), row["distribution"]
            n, accuracy = int(row["string_length"]), float(row["string_accuracy"])
            points = results[key]
            if n in points or not 0 <= accuracy <= 1:
                raise ValueError((key, n, accuracy))
            points[n] = accuracy
    for key, points in results.items():
        if tuple(sorted(points)) != LENGTHS:
            raise ValueError(f"Incomplete evaluation grid: {key}")
    return results


def aggregate_series(project, condition):
    model = project.removeprefix("bobguo__baseline_")
    condition = condition.removeprefix("test-")
    results = load_results()
    values = [[results[model, seed, condition][n] for seed in SEEDS] for n in LENGTHS]
    return list(LENGTHS), [mean(v) for v in values], [stdev(v) for v in values]
