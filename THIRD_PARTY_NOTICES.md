# Third-Party Notices

## Hugging Face Transformers

The GPT-NeoX model classes in `synthetic_exps/models/` adapt components from
[Hugging Face Transformers](https://github.com/huggingface/transformers).
The shared attention and rotary helpers retain components from these classes.
The ALiBi bias and slope code adapts the Transformers MPT implementation.

Upstream copyright notices:

```text
Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
```

These components are distributed under the
[Apache License, Version 2.0](licenses/Apache-2.0.txt). The upstream license
also includes the notice `Copyright 2018- The Hugging Face team. All rights
reserved.`

Upstream source references:

- [GPT-NeoX model](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py)
- [MPT model and ALiBi bias](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/mpt/modeling_mpt.py)
- [Transformers license](https://github.com/huggingface/transformers/blob/v4.40.0/LICENSE)

The local adaptations implement the copying task's positional variants,
prime-period value/output rotations, shared tensor helpers, and training
interfaces. The affected files retain upstream copyright and license notices
and identify the local modifications. `SOURCE_MANIFEST.json` records the
project source snapshot and release-file checksums.
