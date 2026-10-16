# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# See ../../licenses/Apache-2.0.txt for the license and warranty disclaimer.
# Modified for the copying experiments; see ../../THIRD_PARTY_NOTICES.md.
"""GPT-NeoX with per-head prime-period value/output rotations.

Queries and keys use content-only attention scores.
"""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.generation import GenerationMixin
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel
from .mlp import MLP
from .nope import GPTNoPEAttention
from .common import (
    apply_causal_mask,
    apply_rope,
    apply_rope_inverse,
    build_causal_bias,
    causal_mask,
    get_head_rope_embeddings,
    make_head_primes,
    merge_heads,
    split_qkv,
)

logger = logging.get_logger(__name__)


class GPTNoPENewAttention(nn.Module):
    """NoPE attention with value/output rotation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * getattr(config, "rotary_pct", 0.25))
        if self.rotary_ndims % 2 != 0:
            self.rotary_ndims -= 1
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # Add temperature parameter for attention scaling
        self.attention_temperature = getattr(config, 'attention_temperature', 1.0)

        # Per-head prime periods p_i (start at 5)
        num_primes = getattr(config, 'num_primes', None)
        heads_per_prime = getattr(config, 'heads_per_prime', None)
        head_primes = make_head_primes(
            self.num_attention_heads,
            num_primes=num_primes,
            heads_per_prime=heads_per_prime,
        )
        self.register_buffer("head_primes", head_primes, persistent=False)
        self.register_buffer(
            "head_freqs",
            torch.where(
                head_primes.float() > 1,
                2.0 * torch.pi / head_primes.float(),
                torch.zeros_like(head_primes, dtype=torch.float32),
            ),
            persistent=False,
        )
        self.num_prime_heads = int((head_primes > 1).sum().item())
        
    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            build_causal_bias(max_positions, device=device),
            persistent=False,
        )

    def _get_rope_embeddings(self, position_ids: torch.LongTensor, device: torch.device):
        """Compute per-head prime-cyclic rotation embeddings."""
        return get_head_rope_embeddings(position_ids, self.head_freqs, device=device)

    def _apply_rope(self, tensor: torch.Tensor, rope) -> torch.Tensor:
        return apply_rope(tensor, rope=rope)

    def _apply_rope_inverse(self, tensor: torch.Tensor, rope) -> torch.Tensor:
        return apply_rope_inverse(tensor, rope=rope)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (num_heads * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        query, key, value = split_qkv(qkv, self.num_attention_heads, self.head_size)

        # Split into rotary and pass-through parts
        if self.rotary_ndims > 0:
            # NO ROTATION FOR QUERY AND KEY
            # Just keep them as is (effectively NoPE for Q/K)
            
            # Prepare pre-rotated values: V_pre = R^{-1} * V
            # We want V to be "pre-rotated" such that when "post-rotated" by attention (which adds position), 
            # we effectively get relative position.
            
            rope = self._get_rope_embeddings(position_ids, value.device)
            rope_cache = rope
            
            value_rot = value[..., : self.rotary_ndims]
            value_pass = value[..., self.rotary_ndims :]
            
            # Apply inverse rotation to Value
            value_pre_rot = self._apply_rope_inverse(value_rot, rope)
            
            value_pre = torch.cat((value_pre_rot, value_pass), dim=-1)

        else:
            # no rotary dims, keep V as-is
            value_pre = value
            rope_cache = None

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value_pre = torch.cat((past_value, value_pre), dim=-2)
        present = (key, value_pre) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(
            query,
            key,
            value_pre,
            attention_mask,
            head_mask,
            hidden_states,
            position_ids,
            rope_cache=rope_cache,
        )

        # Reshape outputs
        attn_output = merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None, hidden_states=None, position_ids=None, rope_cache=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal = causal_mask(self, key_length, query_length, key.device)

        # Standard scaled dot-product attention logits
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.norm_factor

        attn_scores = apply_causal_mask(attn_scores, causal)
        
        if attention_mask is not None:
            # apply attention mask (padding mask)
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores / self.attention_temperature, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        # Post-rotate context by current query positions on rotary dims
        if self.rotary_ndims > 0:
            ctx_rot = attn_output[..., : self.rotary_ndims]
            ctx_pass = attn_output[..., self.rotary_ndims :]
            if position_ids is not None:
                pos_q = position_ids[:, -query_length:]
            else:
                start = key_length - query_length
                pos_q = torch.arange(start, start + query_length, device=attn_output.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            
            if rope_cache is not None and position_ids is not None and position_ids.size(-1) == query_length:
                rope = rope_cache
            else:
                rope = self._get_rope_embeddings(pos_q, attn_output.device)
            
            ctx_rot = self._apply_rope(ctx_rot, rope)
            attn_output = torch.cat((ctx_rot, ctx_pass), dim=-1)

        return attn_output, attn_weights



class GPTNoPENewLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        # Use layer_norm parameter to control whether to use layer normalization
        self.layer_norm = getattr(config, 'layer_norm', True)
        if self.layer_norm:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = None
            self.post_attention_layernorm = None
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        
        # First `modified_layers` layers use NoPE New attention, subsequent layers use NoPE attention
        modified_layers = getattr(config, 'modified_layers', 1)
        if layer_idx < modified_layers:
            self.attention = GPTNoPENewAttention(config)
        else:
            self.attention = GPTNoPEAttention(config)
        
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Apply input layer norm if enabled
        input_states = self.input_layernorm(hidden_states) if self.layer_norm else hidden_states
        attention_layer_outputs = self.attention(
            input_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            post_attn_states = self.post_attention_layernorm(hidden_states) if self.layer_norm else hidden_states
            mlp_output = self.mlp(post_attn_states)
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            post_attn_states = self.post_attention_layernorm(attn_output) if self.layer_norm else attn_output
            mlp_output = self.mlp(post_attn_states)
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output
        
        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class GPTVO_1_NoScalarForCausalLM(GPTNeoXPreTrainedModel, GenerationMixin):
    """NoPE model with value/output rotation."""
    # _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNoPENewLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        # Use layer_norm parameter to control final layer norm
        self.layer_norm = getattr(config, 'layer_norm', True)
        if self.layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.final_layer_norm = None

        self.gradient_checkpointing = False

        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value
    
    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        # Expand to batch size if needed for per-token/head rotation
        if position_ids.dim() == 2 and position_ids.size(0) == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        # Apply final layer norm if enabled
        if self.layer_norm:
            hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.embed_out(hidden_states)

        lm_loss = None

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + (hidden_states, presents, all_hidden_states, all_attentions)
            return tuple(v for v in output if v is not None)

        return {
            'logits': lm_logits,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        input_shape = input_ids.shape
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
