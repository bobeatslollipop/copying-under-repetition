"""Evaluate a saved copying model on the manuscript's three conditions."""

import argparse
import csv
import json
import os
from pathlib import Path
import random


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing args.json and latest/")
    parser.add_argument("--min-length", type=int, help="Default: 1 for baselines, 8 for VO")
    parser.add_argument("--max-length", type=int, help="Inclusive; default: 2993 for baselines, 10000 for VO")
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Evaluation RNG seed, separate from training seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args(argv)
    saved = json.loads((cli.run_dir / "args.json").read_text())
    vo = saved["model"] == "VO_1_no_scalar"
    first = cli.min_length if cli.min_length is not None else (8 if vo else 1)
    last = cli.max_length if cli.max_length is not None else (10000 if vo else 2993)
    if first < 1 or last < first or cli.step < 1:
        parser.error("Require 1 <= min-length <= max-length and step >= 1")
    saved.update(min_eval_len=first, max_eval_len=last + 1, eval_step_size=cli.step,
                 eval_batch_size=1, eval_num_batches=3, eval_repeats=32,
                 eval_switch_lower=None, eval_switch_upper=None)
    lengths = list(range(first, last + 1, cli.step))
    if cli.dry_run:
        print(json.dumps({"model":saved["model"], "training_seed":saved.get("seed"),
                          "evaluation_seed":cli.seed, "min_length":first, "max_length":last,
                          "length_count":len(lengths), "examples_per_length":96,
                          "checkpoint":"latest", "conditions":["iid","L100","identical"]}, indent=2))
        return
    if cli.output.exists():
        raise FileExistsError(f"Refusing to overwrite {cli.output}")
    os.environ["WANDB_MODE"] = cli.wandb_mode
    import numpy as np
    import torch
    import wandb
    from .utils.data_utils import get_tokenizer
    from .utils.model_utils import load_model
    from .utils.test_utils import evaluation

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    args = argparse.Namespace(**saved)
    tokenizer, tokens, _ = get_tokenizer(args)
    model = load_model(str(cli.run_dir / "latest"), args, tokenizer)
    model.eval()
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "seed", "evaluation_seed", "distribution",
                                                    "string_length", "string_accuracy", "character_accuracy"))
        writer.writeheader()
        for condition, parameter in (("iid",0), ("L100",100), ("identical",float("inf"))):
            args.eval_L = parameter
            wandb.init(project="copy_evaluation", name=f"{cli.run_dir.name}-{condition}",
                       config=vars(args), reinit=True)
            try:
                accuracies, char_accuracies = evaluation(args, model, tokenizer, tokens, model_type="latest")
                for length, accuracy, char_accuracy in zip(lengths, accuracies, char_accuracies):
                    writer.writerow(dict(model=args.model, seed=saved.get("seed", ""),
                                         evaluation_seed=cli.seed, distribution=condition,
                                         string_length=length, string_accuracy=accuracy,
                                         character_accuracy=char_accuracy))
                stream.flush()
            finally:
                wandb.finish()


if __name__ == "__main__":
    main()
