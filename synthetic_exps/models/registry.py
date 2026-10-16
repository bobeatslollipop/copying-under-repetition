from .alibi import GPTNeoXAlibiForCausalLM
from .hard_alibi import GPTNeoXHardAlibiForCausalLM
from .nope import GPTNeoXNoPEForCausalLM
from .rope import GPTNeoXRoPEForCausalLM
from .sinusoidal import GPTNeoXSinusoidalForCausalLM
from .VO_1_no_scalar import GPTVO_1_NoScalarForCausalLM


MODEL_MAPPING = {
    "nope": GPTNeoXNoPEForCausalLM,
    "rope": GPTNeoXRoPEForCausalLM,
    "alibi": GPTNeoXAlibiForCausalLM,
    "hard_alibi": GPTNeoXHardAlibiForCausalLM,
    "sinusoidal": GPTNeoXSinusoidalForCausalLM,
    "VO_1_no_scalar": GPTVO_1_NoScalarForCausalLM,
}
ROTARY_PCT_MODELS = {"rope", "alibi", "hard_alibi", "VO_1_no_scalar"}


def normalize_model_name(name):
    return name.removeprefix("T_")


def get_model_class(name):
    name = normalize_model_name(name)
    if name not in MODEL_MAPPING:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_MAPPING)}")
    return name, MODEL_MAPPING[name]


def apply_model_specific_config(model_name, args, config_kwargs):
    if model_name == "hard_alibi" and args.num_masked_heads is not None:
        config_kwargs["num_masked_heads"] = args.num_masked_heads
    if model_name == "sinusoidal":
        config_kwargs["sinusoidal_scale"] = getattr(args, "sinusoidal_scale", None)
    return config_kwargs


def needs_rotary_pct(model_name):
    return model_name in ROTARY_PCT_MODELS
