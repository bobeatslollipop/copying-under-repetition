# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# See ../../../licenses/Apache-2.0.txt for the license and warranty disclaimer.
# Modified for the copying experiments; see ../../../THIRD_PARTY_NOTICES.md.
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoPEEmbeddings:
    sin: torch.Tensor
    cos: torch.Tensor
    layout: str


def build_rope_inv_freq(rotary_ndims):
    if rotary_ndims <= 0:
        return torch.empty(0, dtype=torch.float32)
    return 1.0 / (10000 ** (torch.arange(0, rotary_ndims, 2).float() / rotary_ndims))


def get_rope_embeddings(position_ids: torch.LongTensor, inv_freq: torch.Tensor, device=None):
    """Build standard RoPE embeddings as a named object, never a bare tuple."""
    if device is None:
        device = position_ids.device
    inv_freq = inv_freq.to(device)
    position_ids = position_ids.to(device)
    if position_ids.dim() == 1:
        sinusoid_inp = torch.einsum("s,d->sd", position_ids.float(), inv_freq)
        return RoPEEmbeddings(
            sin=torch.sin(sinusoid_inp),
            cos=torch.cos(sinusoid_inp),
            layout="seq",
        )
    if position_ids.dim() == 2:
        sinusoid_inp = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
        return RoPEEmbeddings(
            sin=torch.sin(sinusoid_inp),
            cos=torch.cos(sinusoid_inp),
            layout="batch_seq",
        )
    raise ValueError(f"position_ids must be rank 1 or 2, got rank {position_ids.dim()}")


def get_head_rope_embeddings(position_ids: torch.LongTensor, head_freqs: torch.Tensor, device=None):
    """Build per-head one-frequency RoPE embeddings with shape [B, S, H, 1]."""
    if device is None:
        device = position_ids.device
    position_ids = position_ids.to(device)
    if position_ids.dim() != 2:
        raise ValueError("head RoPE embeddings require position_ids with shape [batch, seq]")
    theta = position_ids.float().unsqueeze(-1) * head_freqs.to(device).view(1, 1, -1)
    return RoPEEmbeddings(
        sin=torch.sin(theta).unsqueeze(-1),
        cos=torch.cos(theta).unsqueeze(-1),
        layout="batch_seq_head",
    )


def get_head_scaled_rope_embeddings(
    position_ids: torch.LongTensor,
    inv_freq: torch.Tensor,
    head_scales: torch.Tensor,
    device=None,
):
    """Build standard RoPE embeddings multiplied by a per-head scale/selector."""
    if device is None:
        device = position_ids.device
    position_ids = position_ids.to(device)
    if position_ids.dim() != 2:
        raise ValueError("head-scaled RoPE requires position_ids with shape [batch, seq]")
    theta_base = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq.to(device))
    theta = theta_base.unsqueeze(2) * head_scales.to(device).view(1, 1, -1, 1)
    return RoPEEmbeddings(
        sin=torch.sin(theta),
        cos=torch.cos(theta),
        layout="batch_seq_head",
    )


def _align_rope(rope: RoPEEmbeddings):
    if rope.layout == "seq":
        return rope.sin.unsqueeze(0).unsqueeze(0), rope.cos.unsqueeze(0).unsqueeze(0)
    if rope.layout == "batch_seq":
        return rope.sin.unsqueeze(1), rope.cos.unsqueeze(1)
    if rope.layout == "batch_seq_head":
        return rope.sin.permute(0, 2, 1, 3), rope.cos.permute(0, 2, 1, 3)
    raise ValueError(f"Unsupported RoPE layout: {rope.layout}")


def _apply_rope_with_sin_cos(tensor: torch.Tensor, *, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x_even = tensor[..., ::2]
    x_odd = tensor[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


def apply_rope(tensor: torch.Tensor, *, rope: RoPEEmbeddings) -> torch.Tensor:
    sin, cos = _align_rope(rope)
    return _apply_rope_with_sin_cos(tensor, sin=sin, cos=cos)


def apply_rope_inverse(tensor: torch.Tensor, *, rope: RoPEEmbeddings) -> torch.Tensor:
    sin, cos = _align_rope(rope)
    return _apply_rope_with_sin_cos(tensor, sin=-sin, cos=cos)


def resolve_sinusoidal_scale(config):
    sinusoidal_scale = getattr(config, "sinusoidal_scale", None)
    if sinusoidal_scale is None:
        sinusoidal_scale = getattr(config, "initializer_range", 0.02)
    return float(sinusoidal_scale)


def build_sinusoidal_position_embeddings(
    max_positions: int,
    hidden_size: int,
    device=None,
    dtype=None,
    scale: float = 1.0,
):
    """Build fixed absolute sinusoidal PE with width d_model."""
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    pe = torch.zeros(max_positions, hidden_size, device=device, dtype=torch.float32)
    position = torch.arange(max_positions, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_size)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    pe = pe * float(scale)
    if dtype is not None:
        pe = pe.to(dtype=dtype)
    return pe
