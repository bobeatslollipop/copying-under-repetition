# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# See ../../licenses/Apache-2.0.txt for the license and warranty disclaimer.
# Modified for the copying experiments; see ../../THIRD_PARTY_NOTICES.md.
"""GPT-NeoX with finite-window hard attention heads."""

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

from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel
from .mlp import MLP
from .common import (
    apply_causal_mask,
    build_causal_bias,
    build_hard_alibi_masks,
    causal_mask,
    merge_heads,
    scaled_attention_scores,
    split_qkv,
)

logger = logging.get_logger(__name__)


class GPTHardAlibiAttention(nn.Module):
    """
    Multi-head attention module with hard alibi masking for specific attention heads.
    
    This implementation applies hard alibi patterns to the first `num_masked_heads` 
    attention heads, where each masked head can only attend to positions within 
    a specific offset range. This is designed to improve performance on tasks 
    that require copying or repetition patterns by constraining attention patterns.
    """

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
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # Add temperature parameter for attention scaling
        self.attention_temperature = getattr(config, 'attention_temperature', 1.0)
        
        # Cache frequently used values
        # Clamp masked heads to available heads to avoid shape/broadcast issues
        self.num_masked_heads = min(getattr(config, 'num_masked_heads', self.num_attention_heads // 2), self.num_attention_heads)
        self.use_parallel_residual = getattr(config, 'use_parallel_residual', False)
        
        # Pre-compute hard alibi masks for efficiency
        self._init_hard_alibi_masks(config.max_position_embeddings)

    def _init_bias(self, max_positions, device=None):
        """
        Initialize the causal mask bias buffer for attention computation.
        
        Args:
            max_positions (int): Maximum sequence length for the causal mask
            device (torch.device, optional): Device to place the bias tensor on
        """
        self.register_buffer(
            "bias",
            build_causal_bias(max_positions, device=device),
            persistent=False,
        )
    
    def _init_hard_alibi_masks(self, max_positions):
        """
        Pre-compute hard alibi masks for all possible head configurations.
        This avoids computing masks in the forward pass.
        """
        num_masked_heads = self.num_masked_heads
        if num_masked_heads == 0:
            self.register_buffer(
                "hard_alibi_masks", 
                torch.empty(0, max_positions, max_positions, dtype=torch.bool),
                persistent=False
            )
            return
            
        masks = build_hard_alibi_masks(num_masked_heads, max_positions)
        self.register_buffer("hard_alibi_masks", masks, persistent=False)

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
        """
        Forward pass for the hard alibi attention module.
        
        Args:
            hidden_states (torch.FloatTensor): Input hidden states of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.FloatTensor): Attention mask to prevent attention to certain positions
            position_ids (torch.LongTensor): Position indices for rotary embeddings
            head_mask (torch.FloatTensor, optional): Optional mask for attention heads
            layer_past (Tuple[torch.Tensor], optional): Cached key/value states from previous forward passes
            use_cache (bool, optional): Whether to return cached key/value states for generation
            output_attentions (bool, optional): Whether to return attention weights
            
        Returns:
            Tuple: Contains attention output, present key/value states (if use_cache=True), 
                   and attention weights (if output_attentions=True)
        """
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)
        
        batch_size = qkv.shape[0]
        query, key, value = split_qkv(qkv, self.num_attention_heads, self.head_size)

        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        
        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape and project outputs
        attn_output = merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Compute attention scores with hard alibi masking for specific heads.
        
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, num_attention_heads, seq_len, attn_head_size]
            key (torch.Tensor): Key tensor of shape [batch_size, num_attention_heads, seq_len, attn_head_size]
            value (torch.Tensor): Value tensor of shape [batch_size, num_attention_heads, seq_len, attn_head_size]
            attention_mask (torch.Tensor, optional): Optional attention mask for padding tokens
            head_mask (torch.Tensor, optional): Optional mask for specific attention heads
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and attention weights
        """
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal = causal_mask(self, key_length, query_length, key.device)

        attn_scores = scaled_attention_scores(query, key, self.norm_factor)
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = apply_causal_mask(attn_scores, causal)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        # Apply hard alibi masking
        if self.num_masked_heads > 0 and hasattr(self, 'hard_alibi_masks'):
            # Use pre-computed masks for efficiency
            seq_len = attn_scores.shape[-1]
            masked_heads = min(self.num_masked_heads, num_attention_heads)
            if seq_len <= self.hard_alibi_masks.shape[-1] and masked_heads > 0:
                # Use pre-computed masks
                query_start = key_length - query_length
                alibi_masks = self.hard_alibi_masks[
                    :masked_heads,
                    query_start:key_length,
                    :key_length,
                ]
                alibi_masks = alibi_masks.unsqueeze(0)  # [1, masked_heads, q, k]
                
                # Apply masks: set masked positions to mask_value
                masked_positions = ~alibi_masks
                attn_scores[:, :masked_heads] = torch.where(
                    masked_positions,
                    mask_value,
                    attn_scores[:, :masked_heads]
                )
            else:
                # Fallback to dynamic computation for longer sequences
                self._apply_dynamic_hard_alibi(attn_scores, mask_value, masked_heads)
        
        attn_scores = attn_scores / self.attention_temperature
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
    
    def _apply_dynamic_hard_alibi(self, attn_scores, mask_value, num_masked_heads):
        """
        Apply hard alibi masking dynamically for sequences longer than pre-computed masks.
        This is a fallback method for very long sequences.
        """
        _, _, query_length, key_length = attn_scores.shape
        query_start = key_length - query_length
        alibi_masks = build_hard_alibi_masks(
            num_masked_heads,
            key_length,
            device=attn_scores.device,
        )[:, query_start:key_length, :key_length]
        attn_scores[:, :num_masked_heads] = attn_scores[:, :num_masked_heads].masked_fill(
            ~alibi_masks.unsqueeze(0),
            mask_value,
        )


class GPTNeoXHardAlibiLayer(nn.Module):
    """
    A single transformer layer with hard alibi attention and MLP components.
    """

    def __init__(self, config):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.layer_norm = getattr(config, 'layer_norm', True)
        
        if self.layer_norm:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = None
            self.post_attention_layernorm = None
            
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPTHardAlibiAttention(config)
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
        # Attention computation (optimized)
        input_states = self.input_layernorm(hidden_states) if self.input_layernorm is not None else hidden_states
        attention_layer_outputs = self.attention(
            input_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]
        present = attention_layer_outputs[1] if use_cache else None
        attn_weights = attention_layer_outputs[2] if output_attentions else None
        
        # Apply dropout
        attn_output = self.post_attention_dropout(attn_output)

        if self.use_parallel_residual:
            # Parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
            post_attn_states = self.post_attention_layernorm(hidden_states) if self.post_attention_layernorm is not None else hidden_states
            mlp_output = self.mlp(post_attn_states)
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = hidden_states + attn_output + mlp_output
        else:
            # Sequential residual: x = x + attn(ln1(x)) + mlp(ln2(x + attn(ln1(x))))
            hidden_states = hidden_states + attn_output
            post_attn_states = self.post_attention_layernorm(hidden_states) if self.post_attention_layernorm is not None else hidden_states
            mlp_output = self.mlp(post_attn_states)
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = hidden_states + mlp_output

        # Prepare outputs
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present,)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs


class GPTNeoXHardAlibiModel(GPTNeoXPreTrainedModel):
    """
    The main GPTNeoX model with hard alibi attention, consisting of embedding layer,
    multiple transformer layers, and final layer normalization.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXHardAlibiLayer(config) for _ in range(config.num_hidden_layers)])
        
        self.layer_norm = getattr(config, 'layer_norm', True)
        if self.layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.final_layer_norm = None

        self.gradient_checkpointing = False
        

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

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
        
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            batch_size, seq_length = input_ids.size()
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Handle past key values 
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        # Generate position IDs if not provided
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, 
                dtype=torch.long, device=device
            ).unsqueeze(0)

        # Process attention mask
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask.to(dtype=self.dtype)) * torch.finfo(self.dtype).min

        # Prepare head mask
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)
        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Initialize output collections
        presents = [] if use_cache else None
        all_attentions = [] if output_attentions else None
        all_hidden_states = [] if output_hidden_states else None
        
        # Process layers
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            request_attentions = output_attentions

            # Layer forward pass
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    request_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=request_attentions,
                )
            
            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])
            if output_attentions:
                all_attentions.append(outputs[2 if use_cache else 1])
            

        # Final layer norm and output preparation
        if self.layer_norm and self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Convert lists to tuples for output
        if use_cache:
            presents = tuple(presents)
        if output_attentions:
            all_attentions = tuple(all_attentions)
        if output_hidden_states:
            all_hidden_states = tuple(all_hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTNeoXHardAlibiForCausalLM(GPTNeoXPreTrainedModel):
    """
    GPTNeoX model with hard alibi attention for causal language modeling.
    This model adds a language modeling head on top of the base model.
    """

    # _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXHardAlibiModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        # Compute loss if labels provided
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # Shift for next-token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss efficiently
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation by handling past key values and position ids.
        
        Args:
            input_ids (torch.LongTensor): Input token ids
            past_key_values (Tuple, optional): Cached key-value states from previous generation steps
            attention_mask (torch.LongTensor, optional): Attention mask for padding tokens
            inputs_embeds (torch.FloatTensor, optional): Input embeddings (used only in first generation step)
            **kwargs: Additional keyword arguments including position_ids
            
        Returns:
            Dict: Dictionary containing prepared model inputs
        """
        # Handle past key values for generation
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
            else:
                input_ids = input_ids[:, -1:]

        # Generate position IDs for generation
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # Prepare model inputs for generation
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        
        model_inputs = {
            "input_ids": input_ids if inputs_embeds is None or past_key_values is not None else None,
            "inputs_embeds": inputs_embeds if inputs_embeds is not None and past_key_values is None else None,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorder cached key-value states for beam search.
        
        Args:
            past_key_values (Tuple): Cached key-value states from all layers
            beam_idx (torch.LongTensor): Beam indices for reordering
            
        Returns:
            Tuple: Reordered cached key-value states
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
