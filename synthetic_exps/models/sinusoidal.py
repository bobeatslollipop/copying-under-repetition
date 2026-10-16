# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# See ../../licenses/Apache-2.0.txt for the license and warranty disclaimer.
# Modified for the copying experiments; see ../../THIRD_PARTY_NOTICES.md.
"""GPTNeoX with fixed absolute sinusoidal positional encodings."""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel

from .common import build_sinusoidal_position_embeddings, resolve_sinusoidal_scale
from .nope import GPTNeoXNoPEForCausalLM, GPTNeoXNoPEModel


class GPTNeoXSinusoidalModel(GPTNeoXNoPEModel):
    def __init__(self, config):
        super().__init__(config)
        self.sinusoidal_scale = resolve_sinusoidal_scale(config)
        self.register_buffer(
            "sinusoidal_position_embeddings",
            build_sinusoidal_position_embeddings(
                config.max_position_embeddings,
                config.hidden_size,
                scale=self.sinusoidal_scale,
            ),
            persistent=False,
        )

    def _get_position_embeddings(self, position_ids: torch.LongTensor, dtype: torch.dtype):
        needed_positions = int(position_ids.max().item()) + 1
        if needed_positions > self.sinusoidal_position_embeddings.shape[0]:
            self.sinusoidal_position_embeddings = build_sinusoidal_position_embeddings(
                needed_positions,
                self.config.hidden_size,
                device=position_ids.device,
                dtype=self.sinusoidal_position_embeddings.dtype,
                scale=self.sinusoidal_scale,
            )
        position_embeddings = self.sinusoidal_position_embeddings.to(
            device=position_ids.device,
            dtype=dtype,
        )
        return position_embeddings[position_ids]

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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                seq_length + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        inputs_embeds = inputs_embeds + self._get_position_embeddings(position_ids, inputs_embeds.dtype)

        return super().forward(
            input_ids=None,
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


class GPTNeoXSinusoidalForCausalLM(GPTNeoXNoPEForCausalLM):
    def __init__(self, config):
        GPTNeoXPreTrainedModel.__init__(self, config)
        self.gpt_neox = GPTNeoXSinusoidalModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
