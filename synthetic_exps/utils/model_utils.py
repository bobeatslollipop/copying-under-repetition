import os
import torch
from transformers import GPTNeoXConfig

from ..models.registry import (
    MODEL_MAPPING,
    apply_model_specific_config,
    get_model_class,
    needs_rotary_pct,
    normalize_model_name,
)


def _normalize_model_name(name):
    return normalize_model_name(name)

def get_model(args, tokenizer):
    """
    Instantiate a model based on args.model.
    """
    model_name, model_class = get_model_class(args.model)

    # Transformer models using GPTNeoXConfig
    config_kwargs = {
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.hidden_size * 4,
        "num_attention_heads": args.heads,
        "num_hidden_layers": args.layers,
        "vocab_size": len(tokenizer),
        "hidden_act": getattr(args, 'mlp_act', 'gelu'),
        "torch_dtype": torch.float32,
        "attention_temperature": getattr(args, 'attention_temperature', 1.0),
        "num_primes": getattr(args, 'num_primes', None),
        "heads_per_prime": getattr(args, 'heads_per_prime', None),
        "modified_layers": getattr(args, 'modified_layers', 1),
    }

    config_kwargs = apply_model_specific_config(model_name, args, config_kwargs)
    
    # Pass the 'layer_norm' toggle to all models
    config_kwargs["layer_norm"] = args.layer_norm
        
    config = GPTNeoXConfig(**config_kwargs)

    # Post-initialization config updates
    # Always force rotary_pct — hasattr is unreliable in transformers >= 5.4.0
    # where rotary_pct was removed from GPTNeoXConfig
    if needs_rotary_pct(model_name):
        config.rotary_pct = 0.5
    



    model = model_class(config)

    # Ensure model is float32 and on correct device
    model = model.to(dtype=torch.float32)
    return model


def load_model(model_path, args, tokenizer):
    """
    Load a previously saved model from the specified path.

    We deliberately bypass `from_pretrained` because newer transformers versions
    silently mutate the loaded config (e.g. syncing `rotary_pct` into
    `rope_parameters.partial_rotary_factor`), which changes forward behavior even
    for models that don't use RoPE (e.g. alibi). Instead we build a fresh model
    via `get_model` — which matches the training-time config exactly — and then
    load the raw state dict directly.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if not os.path.isdir(model_path):
        raise ValueError(f"Expected directory for transformer model, got: {model_path}")

    get_model_class(args.model)

    # Build a fresh model with the same config as training
    model = get_model(args, tokenizer)

    # Load the saved state dict (prefer safetensors, fall back to pytorch_model.bin)
    sd_path_safe = os.path.join(model_path, 'model.safetensors')
    sd_path_bin = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(sd_path_safe):
        from safetensors.torch import load_file
        saved_sd = load_file(sd_path_safe)
    elif os.path.exists(sd_path_bin):
        saved_sd = torch.load(sd_path_bin, map_location='cpu', weights_only=False)
    else:
        raise FileNotFoundError(
            f"Neither model.safetensors nor pytorch_model.bin found in {model_path}"
        )

    # Handle fp8 parametrized weights: map back to regular weight names
    if getattr(args, 'precision', None) == 'fp8':
        fixed_sd = {}
        for k, v in saved_sd.items():
            if k.endswith('.parametrizations.weight.original'):
                fixed_sd[k.replace('.parametrizations.weight.original', '.weight')] = v
            else:
                fixed_sd[k] = v
        saved_sd = fixed_sd

    result = model.load_state_dict(saved_sd, strict=False)
    if result.missing_keys:
        print(f"Warning: missing keys during load_state_dict: {result.missing_keys[:10]}")
    if result.unexpected_keys:
        print(f"Warning: unexpected keys during load_state_dict: {result.unexpected_keys[:10]}")

    # Move model to GPU if available and enforce float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device, dtype=torch.float32)

    if getattr(args, 'precision', None) == 'fp8':
        from .train_utils import apply_fp8_simulation
        apply_fp8_simulation(model)
        print("Applied simulated FP8 forward pass to loaded model and restored exact parameters.")

    print(f"Successfully loaded model from: {model_path}")
    return model
