# Copying Under Repetition: From Prefix Matching to Position Hashing

Anxin Guo, Tianlong Huang, Chenxiao Yang, and Zhiyuan Li

[Paper](https://arxiv.org/abs/2610.26722) | [PDF](https://arxiv.org/pdf/2610.26722)

Code, experimental data, and raw model responses accompanying our paper.
The synthetic models are NoPE, RoPE, ALiBi, Hard-ALiBi, Sinusoidal, and VO-Rotation.
The LLM data cover the six model variants plotted in the paper.

VO-Rotation uses periods **5, 7, 11, 13**, whose product is **5005**.
The synthetic model source is pinned to commit
`8d3195b11f39743f9eae89d4c07810ca96de7f80`. Portable entry points provide
training, checkpoint loading, and evaluation from this directory.

## Contents

```text
configs/             Recorded training presets and LLM variant settings
data/llm/            Six original per-trial CSVs, including raw model replies
data/synthetic/      Baseline/VO evaluations and the seed-42 training loss
llm/                 API clients, string sampling, extraction, and scoring
synthetic_exps/      Six model implementations, training, and evaluation
plots/               Scripts for the five figures shown in the manuscript
tests/               Recorded-data and LLM scoring checks
DATA_MANIFEST.json   Dataset counts and SHA-256 checksums
SOURCE_MANIFEST.json Source revision, original hashes, and release-file hashes
CITATION.cff         Paper citation metadata
THIRD_PARTY_NOTICES.md Upstream code attribution and license references
LICENSE              MIT license for project code and bundled data
```

The recorded result tables regenerate all reported figures locally.
Training and LLM commands write new results under `outputs/`.

## Setup

Run commands from this directory. Plotting and LLM experiments require Python
3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Synthetic training requires Python 3.11 or newer and a suitable PyTorch GPU
installation. `requirements-training.txt` records the core package versions
from the archived training/evaluation environment:

```bash
python -m pip install -r requirements-training.txt
```

The training environment pins Transformers for the custom GPT-NeoX models and
their rotary configuration. The lighter `requirements.txt` environment supports
dataset validation, figure regeneration, and LLM experiments.

## Reproduce the Figures

```bash
python plots/plot_llm_main.py
python plots/plot_synthetic_test_accuracy.py
python plots/plot_synthetic_training_trajectory.py
python plots/plot_vo_generalization_10000.py
```

These commands write to `figures/`:

| Script | Manuscript figure |
| --- | --- |
| `plot_llm_main.py` | Frontier-LLM copying accuracy across repetition conditions |
| `plot_synthetic_test_accuracy.py` | Standard positional baselines and VO near the training range |
| `plot_synthetic_training_trajectory.py` | VO training loss, seed 42 |
| `plot_vo_generalization_10000.py` | VO extrapolation to length 10,000 and per-seed curves |

All accuracy plots and numerical summaries use the recorded pointwise
accuracies. The main VO extrapolation plot shows the three-seed mean on the
complete evaluation grid. Shaded bands span one sample standard deviation
across seeds, clipped to the accuracy range `[0, 1]`.

## Train the Synthetic Models

The JSON presets contain the full experiment settings. Both families train on
source lengths sampled uniformly from 1 through 2000. The training mixture is
10% i.i.d., 10% all-identical, and 80% geometric-block strings. For the block
component, `L` is uniform on `[0, 300]` and the geometric parameter is
`1 / (L + 1)`. Each block's letter is independently uniform over 26 lowercase
letters; neighboring sampled blocks can therefore share a letter.

| Setting | Standard positional baselines | VO-Rotation |
| --- | --- | --- |
| Layers / width / heads | 6 / 512 / 8 | 2 / 256 / 4 |
| Precision | bf16 | Simulated FP8 linear inputs and weights |
| Microbatch / accumulation | 4 / 8 | 8 / 4 |
| Effective batch size | 32 | 32 |
| Maximum microsteps | 2,560,000 | 1,600,000 |
| Learning rate / weight decay | 0.0001 / 0 | 0.0001 / 0 |
| Scheduler | Warmup-stable-decay | Warmup-stable-decay |
| Warmup optimizer updates | 2000 | 2000 |
| Cooldown fraction | 0.3 | 0.3 |
| Cooldown trigger | EMA improvement plateau | Training-loss EMA below 0.0001 |

Each VO head rotates 32 of its 64 value coordinates, forming 16 two-dimensional
blocks. All blocks in a head share that head's single prime-period frequency.

The scheduler can finish before the maximum microstep budget. Plateau-detector
settings are in the JSON files. The original runs used one RTX A6000 per
baseline and one RTX A4000 per VO model.

Run all baseline seeds:

```bash
for model in nope rope alibi hard_alibi sinusoidal; do
  for seed in 42 43 44; do
    python -m synthetic_exps.train --model "$model" --seed "$seed"
  done
done
```

Run the VO seeds:

```bash
for seed in 42 43 44; do
  python -m synthetic_exps.train --model VO_1_no_scalar --seed "$seed"
done
```

Inspect a resolved configuration without loading ML packages or training:

```bash
python -m synthetic_exps.train --model VO_1_no_scalar --seed 42 --dry-run
```

Checkpoints are saved under
`outputs/checkpoints/copy_<model>/seed_<seed>_training/`, with `args.json`,
`checkpoint.pt`, `latest/`, and `ema/`. Repeating the same command resumes an
existing checkpoint. Use a new `--checkpoint-root` for a separate run. W&B is
disabled by default; `--wandb-mode offline` enables local W&B logs and
`--wandb-mode online` enables account-backed logging.

The task is encoded as `$<source>;:<answer>.`. The loss covers the answer
including its final period. The historical vocabulary is retained, including
unused digit/operator tokens, so model dimensions remain unchanged.

## Evaluate a Trained Checkpoint

Evaluate the final `latest` checkpoint, as in the paper:

```bash
python -m synthetic_exps.evaluate \
  --run-dir outputs/checkpoints/copy_nope/seed_42_training \
  --output outputs/evaluation/nope_seed42.csv

python -m synthetic_exps.evaluate \
  --run-dir outputs/checkpoints/copy_VO_1_no_scalar/seed_42_training \
  --output outputs/evaluation/vo_seed42.csv
```

Defaults use lengths `1, 9, ..., 2993` for baselines and
`8, 16, ..., 10000` for VO, with 96 examples per length and condition
(batch size 1, three batches, 32 repeats). Conditions are i.i.d.,
geometric blocks with `L=100`, and all-identical strings. Optional
`--min-length`, `--max-length` (inclusive), and `--step` select a smaller grid;
`--dry-run` prints the resolved evaluation setup.

Evaluation uses teacher-forced next-token argmax predictions. String-level
exact-copy accuracy requires every answer position, including the end marker,
to match. `--seed` initializes the random stream for a new evaluation run,
independently of the training seed.

## LLM Experiments

Each recorded variant has 50 trials for each of five lengths
(`30, 100, 300, 1000, 3000`) and three conditions, totaling 750 trials.
The six CSVs contain **4,500 trials**. Empty outputs count as copying failures.
Qwen3.5 returns 112 empty outputs: 43 on geometric-block strings and 69 on
all-identical strings, with none on i.i.d. strings. All trials remain in the
accuracy denominator.

| Variant / CSV stem | Model ID | Reasoning setting |
| --- | --- | --- |
| `397b` | `qwen3.5:397b-cloud` | Thinking disabled |
| `deepseek-v4-pro-no-thinking` | `deepseek-v4-pro` | Thinking disabled |
| `gemini-3.1-pro-low-thinking` | `gemini-3.1-pro-preview` | Low thinking |
| `gemini-3.1-pro-medium-thinking` | `gemini-3.1-pro-preview` | Medium thinking |
| `gpt-5.5` | `gpt-5.5` | No reasoning |
| `gpt-5.5-medium-reasoning` | `gpt-5.5` | Medium reasoning |

Provider credentials are read from `OPENAI_API_KEY`, `GEMINI_API_KEY`, and
`DEEPSEEK_API_KEY`. The Qwen variant uses an authenticated Ollama service at
`http://localhost:11434`; `--base-url` overrides a provider endpoint.

Print settings without API requests:

```bash
python -m llm.run --variant 397b --dry-run
```

Run one variant or all six with the recorded length/condition grid and a
20,000-output-token budget:

```bash
python -m llm.run --variant 397b
python -m llm.run --variant deepseek-v4-pro-no-thinking
python -m llm.run --variant gemini-3.1-pro-low-thinking
python -m llm.run --variant gemini-3.1-pro-medium-thinking
python -m llm.run --variant gpt-5.5
python -m llm.run --variant gpt-5.5-medium-reasoning
```

These commands contact the configured providers and can incur API charges.
They write new CSVs and credential-free settings files under `outputs/llm/`,
separately from the bundled paper data. `--lengths 30 --n-trials 1` selects a
small run. An optional `--seed` controls new target sampling.

`--resume` appends missing trials while preserving every recorded outcome,
including unsuccessful trials. It requires matching settings and a valid,
complete CSV; use a new output directory to change the experiment. With an
explicit seed, the sampler consumes skipped trials' draws so missing targets
match an uninterrupted run with the same settings.

OpenAI, DeepSeek, and Gemini clients retry failed requests with the same target.
After a successful retry, the final response determines the trial's error
status. Generation limits, incomplete responses, and exhausted request retries
remain failures.

### Prompt and Scoring

The system prompt is:

> You are a precise string-copying assistant. You will see a string enclosed between '$' and '.', followed by ':'. Output ONLY the enclosed string, character by character, with absolutely nothing else -- no spaces before or after, no explanation, no newline.

The literal user prompt is `$<target>.  :`, with two spaces before the colon.
The model should return the target without delimiters.

`raw_output` stores the model text, with returned thinking text wrapped in
`<think>` tags where available and line breaks represented as literal `\n`.
`prediction` stores the extracted answer. Extraction removes thinking blocks
and cleans prefixes, delimiters, quotes, backticks, and spaces. Among
lowercase-only lines, it chooses the line closest to the target length;
otherwise it uses the cleaned last line. The metric is extracted-answer exact
match, with target length used for line selection. Empty outputs receive an
empty prediction and score zero. Trials with explicit request or generation
errors retain partial raw text and also receive an empty prediction. All trials
contribute to accuracy. The bundled CSV fields, replies, and scores are preserved.

New Ollama runs also write `<variant>.responses.jsonl` alongside the CSV. It
records the HTTP status, completion marker, stop reason, token counts, and timing
fields returned by the service, keyed by trial and target SHA-256. Stream errors
and missing completion markers are recorded explicitly. Resume appends new
records without rewriting prior outcomes; authentication headers are not stored.

## Checks

```bash
python tests/check_release.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

With training dependencies installed:

```bash
python -m unittest discover -s synthetic_exps/tests -p 'test_*.py' -v
```

The checks cover source/data checksums, English-only release text, credential
patterns, third-party notices, complete evaluation grids, stored exact-match
scores, launch presets, answer masking, shared attention helpers, and LLM
extraction, empty-output scoring, stream-error handling, response logging, and
request retries.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{guo2026copying,
  title         = {Copying Under Repetition: From Prefix Matching to Position Hashing},
  author        = {Anxin Guo and Tianlong Huang and Chenxiao Yang and Zhiyuan Li},
  year          = {2026},
  eprint        = {2610.26722},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.26722},
  url           = {https://arxiv.org/abs/2610.26722}
}
```

## Data Provenance and Licensing

The six LLM CSVs are byte-for-byte copies of the manuscript's source data.
Synthetic tables retain the original accuracy values and run IDs while
flattening exports and dropping archive-path columns. The seed-42 loss table
contains optimizer-update indices and training loss. `DATA_MANIFEST.json`
records the source files, transformations, row counts, and checksums.

Project code and bundled datasets are released under the [MIT License](LICENSE).
Third-party portions of the adapted Transformers components retain their
Apache 2.0 license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`licenses/Apache-2.0.txt`. Dependencies retain their respective licenses.
