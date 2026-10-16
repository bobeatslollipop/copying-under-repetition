"""Train a manuscript model without cluster-specific launch scripts."""

import argparse
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("nope", "rope", "alibi", "hard_alibi", "sinusoidal", "VO_1_no_scalar")


def resolve_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, help="Optional replacement for the family JSON preset")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--steps", type=int, help="Override the microstep budget")
    parser.add_argument("--max-train-len", type=int, help="Override maximum source length")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved parameters without ML imports or training")
    cli = parser.parse_args(argv)
    family = "vo" if cli.model == "VO_1_no_scalar" else "baseline"
    config_path = cli.config or ROOT / "configs/synthetic" / f"{family}.json"
    config = json.loads(config_path.read_text())
    config.update(
        model=cli.model,
        seed=cli.seed,
        checkpoint_root=str(cli.checkpoint_root.resolve()),
        wandb_project=f"copy_{cli.model}",
        wandb_run_name=f"seed_{cli.seed}_training",
        wandb_resume_id=None,
    )
    for key, value in (("steps", cli.steps), ("max_train_len", cli.max_train_len)):
        if value is not None:
            config[key] = value
    if config["task"] != "copy":
        parser.error("Only the manuscript copying task is included")
    if cli.model == "VO_1_no_scalar" and config.get("prime_start", 5) != 5:
        parser.error("The archived paper implementation starts its prime periods at 5")
    if config["steps"] < config["gradient_accumulation_steps"]:
        parser.error("The budget must include at least one optimizer update")
    return cli, argparse.Namespace(**config)


def main(argv=None):
    cli, args = resolve_args(argv)
    if cli.dry_run:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return

    os.environ["WANDB_MODE"] = cli.wandb_mode
    os.environ.setdefault("WANDB_DIR", str(Path("outputs/wandb").resolve()))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import wandb
    from .utils.data_utils import get_tokenizer, get_train_dataset
    from .utils.model_utils import get_model
    from .utils.train_utils import train

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)

    run_dir = Path(args.checkpoint_root) / args.wandb_project / args.wandb_run_name
    saved_path = run_dir / "args.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        for key in ("model", "seed", "hidden_size", "layers", "heads", "precision", "max_train_len"):
            if saved.get(key) != getattr(args, key):
                raise ValueError(f"Existing run differs in {key}; choose a new --checkpoint-root")
    checkpoint = run_dir / "checkpoint.pt"
    init_kwargs = {}
    if checkpoint.exists() and cli.wandb_mode == "online":
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("wandb_run_id"):
            init_kwargs.update(id=state["wandb_run_id"], resume="must")
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), **init_kwargs)
    try:
        tokenizer, tokens, _ = get_tokenizer(args)
        dataset = get_train_dataset(args, tokenizer)
        # The original entry point sampled one display batch before model initialization.
        batch = next(iter(dataset))
        print(f"Example: {batch['input'][0]}")
        model = get_model(args, tokenizer)
        train(args, model, dataset, tokens, tokenizer, num_log_steps=args.num_log_steps,
              max_runtime_seconds=args.max_runtime_seconds)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
