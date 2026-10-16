# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
# Licensed under the Apache License, Version 2.0.
# See ../../../licenses/Apache-2.0.txt for the license and warranty disclaimer.
# Modified for the copying experiments; see ../../../THIRD_PARTY_NOTICES.md.
import math

import torch


def build_causal_bias(max_positions, device=None):
    return torch.tril(
        torch.ones((max_positions, max_positions), dtype=torch.bool, device=device)
    ).view(1, 1, max_positions, max_positions)


def ensure_causal_bias(module, key_length, device):
    if key_length > module.bias.shape[-1]:
        module._init_bias(key_length, device=device)


def causal_mask(module, key_length, query_length, device):
    ensure_causal_bias(module, key_length, device)
    return module.bias[:, :, key_length - query_length : key_length, :key_length]


def apply_causal_mask(attn_scores, mask):
    return attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)


def split_qkv(qkv, num_attention_heads, head_size):
    batch_size, seq_len, _ = qkv.shape
    qkv = qkv.view(batch_size, seq_len, num_attention_heads, 3 * head_size)
    query, key, value = qkv.chunk(3, dim=-1)
    return (
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )


def merge_heads(tensor, num_attention_heads, head_size):
    tensor = tensor.transpose(1, 2).contiguous()
    return tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * head_size)


def scaled_attention_scores(query, key, norm_factor):
    batch_size, num_attention_heads, query_length, head_size = query.shape
    key_length = key.size(-2)
    query = query.reshape(batch_size * num_attention_heads, query_length, head_size)
    key = key.reshape(batch_size * num_attention_heads, key_length, head_size)
    attn_scores = torch.bmm(query, key.transpose(1, 2))
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
    return attn_scores * norm_factor


def build_mpt_alibi_slopes(num_heads, alibi_bias_max=8):
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.float32)
    base = base * (alibi_bias_max / num_heads_power_of_2)
    slopes = 1.0 / torch.pow(2, base)
    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:num_heads]
    return slopes


def build_hard_alibi_masks(num_masked_heads, max_positions, device=None):
    if num_masked_heads == 0:
        return torch.empty(0, max_positions, max_positions, dtype=torch.bool, device=device)

    rows = torch.arange(max_positions, device=device).view(1, max_positions, 1)
    cols = torch.arange(max_positions, device=device).view(1, 1, max_positions)
    max_offsets = torch.arange(num_masked_heads, device=device).view(num_masked_heads, 1, 1)
    offsets = rows - cols
    return (offsets >= 0) & (offsets <= max_offsets)
