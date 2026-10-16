"""Run one of the six LLM variants shown in the manuscript."""

import argparse
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    variants = json.loads((ROOT / "configs/llm.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(variants), required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[30,100,300,1000,3000])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm"))
    parser.add_argument("--seed", type=int, help="Optional seed for newly generated targets")
    parser.add_argument("--base-url", help="Override the provider endpoint")
    parser.add_argument("--resume", action="store_true", help="Append missing trials; retain every recorded outcome")
    parser.add_argument("--dry-run", action="store_true", help="Print settings without contacting an API")
    args = parser.parse_args(argv)
    if args.n_trials < 1 or min(args.lengths) < 1 or args.max_output_tokens < 1:
        parser.error("Trial counts, lengths, and output-token budget must be positive")
    variant = variants[args.variant].copy()
    model = variant.pop("model")
    if args.base_url:
        variant["base_url"] = args.base_url
    metadata = dict(variant=args.variant, model=model, **variant,
                    lengths=args.lengths, L_values=[0,100,"inf"], n_trials=args.n_trials,
                    max_output_tokens=args.max_output_tokens, seed=args.seed)
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    from .core import ExperimentConfig, run_experiment, summarize, print_summary_table
    import numpy as np
    output = args.output_dir.resolve()
    if output == (ROOT / "data/llm").resolve() or (ROOT / "data").resolve() in output.parents:
        parser.error("Use an output directory outside the bundled data directory")
    csv_path = output / f"{args.variant}.csv"
    metadata_path = output / f"{args.variant}.metadata.json"
    if csv_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {csv_path}; use a new directory or --resume")
    if csv_path.exists() and not metadata_path.exists():
        raise ValueError("Existing CSV has no settings record; use a new --output-dir")
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        changed = sorted(key for key in set(previous) | set(metadata)
                         if previous.get(key) != metadata.get(key))
        if changed:
            raise ValueError(f"Existing settings differ in {', '.join(changed)}; use a new --output-dir")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(models=[model], lengths=args.lengths, L_values=[0.0,100.0,float("inf")],
                           n_trials=args.n_trials, max_output_tokens=args.max_output_tokens,
                           output_dir=str(output), run_name=args.variant, resume=args.resume, **variant)
    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    cfg.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    cfg.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not metadata_path.exists():
        with metadata_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, indent=2)+"\n")
    results = run_experiment(cfg)
    print_summary_table(summarize(results))


if __name__ == "__main__":
    main()
