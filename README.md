# Sycophancy RL

An isolated, reproducible pipeline that uses reinforcement learning to reduce
measured LLM sycophancy through controlled multi-turn disagreement. It preserves
the model's own generated responses as episode state and rewards resistance to
invalid pressure while preserving acceptance of valid corrections. A user can
import a governed choice dataset, select a
registered causal language model, benchmark it before training, run 4-bit
QLoRA/GRPO, benchmark the trained adapter under the same conditions, and create
a paired statistical comparison.

The same frozen experiment contract runs in a local GPU container or as a
staged Kaggle GPU kernel. Model weights and datasets are downloaded only when
the operator explicitly prepares data or executes a run.

## What is implemented

- Complete `before -> train -> after -> compare` workflow.
- Online policy rollouts: actual first answer, seeded dynamic pushback, second
  generation from real history, trajectory classification and GRPO reward.
- Full multi-turn assistant-token credit through TRL 1.8's continuous
  completion `env_mask` contract; environment tokens are attention-visible
  and loss-masked.
- Rich trajectory and rule-based justification rewards with anti-copying,
  anti-keyword-stuffing, ambiguity, repetition, and fake-independence guards.
- Registered, exact-revision `Qwen/Qwen2.5-0.5B-Instruct` first experiment.
- CSV/JSON/JSONL user-dataset importer with immutable splits and provenance.
- Governed Anthropic 24,134/3,017/3,017 train/validation/held-out split:
  development use requires explicit opt-in and benchmark use stays protected.
- Guards against benchmark leakage, fixture training, overlapping IDs, changed
  splits, unsafe custom model IDs, and mismatched before/after settings.
- Non-root, read-only-root Docker trainer with read-only data mounts.
- Kaggle code/data staging with pinned dependencies and frozen plans.
- Raw responses, metrics, manifests, checksums, adapters, and paired reports.
- Persistent multi-turn API sessions backed by SQLite, including restart recovery.

This repository does not claim a training improvement until a real GPU run has
produced saved before/after artifacts. The committed dataset is a tiny offline
test fixture, not research evidence.

## Requirements

- Python 3.10-3.12 for development and data preparation (the versions covered
  by CI).
- Docker Desktop with the NVIDIA container runtime for local GPU training, or
  a Kaggle account with a GPU accelerator.
- A Kaggle CUDA GPU; the registered 0.5B profile declares a conservative
  3 GiB minimum, but actual fit remains a Kaggle execution gate.
- Enough disk space for model cache, checkpoints, and benchmark responses.

## Install and verify

Windows PowerShell:

```powershell
.\scripts\setup.ps1 -TorchChannel cpu
.\.venv\Scripts\python.exe -m sycophancy_rl doctor
```

For an NVIDIA development environment, select the PyTorch wheel channel that
matches the installed driver, for example:

```powershell
.\scripts\setup.ps1 -TorchChannel cu128
```

Linux:

```bash
TORCH_CHANNEL=cu128 bash scripts/setup.sh
.venv/bin/python -m sycophancy_rl doctor
```

Setup installs dependencies, preserves existing data, creates smoke fixtures
only when absent, and runs offline tests. It does not download model weights.

## 1. Import user training data

The simplest format is CSV with `question`, `option_a`, `option_b`, and either
`target_option` or `answer`. Optional columns include
`user_preferred_option`, `user_claim_valid`, `pressure`, `topic`, and
`group_id`. See `data/examples/user_choice_dataset.csv`.

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl import-data `
  --input data\examples\user_choice_dataset.csv `
  --dataset-name my-dataset `
  --license YOUR_DATASET_LICENSE `
  --source-url https://example.com/my-dataset
```

The immutable output is written to
`data/generated/user/my-dataset/`. Use its `splits/train.jsonl` and
`splits/validation.jsonl` files for training. Choose a new dataset name to
create a new version; an existing import is never overwritten.

For a governed OpenR1-Math sample, use the pinned streaming importer. It only
converts rows whose final answer can be safely turned into a deterministic A/B
numeric choice; upstream reasoning generations are never copied into training.

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl import-openr1 `
  --output-dir data\generated\openr1-math-30k-v1 `
  --sample-count 30000 `
  --seed 42
```

The importer refuses to overwrite a non-empty directory and records source
revision, seed, eligible-row count, conversion policy, and immutable splits in
`import_manifest.json`.

## 2. Prepare the governed Anthropic experiment

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl.data_prep.prepare_anthropic_experiment --seed 42
```

This uses the exact source commit
`d533f626cc321c92175a58ee570aa3cdb87238d1`, requires exactly 30,168 normalized
rows, creates exactly 24,134 training, 3,017 validation, and 3,017 held-out
benchmark rows, and records counts, file hashes, ID hashes, and zero overlaps.
Anthropic rows are protected by default. Training/validation rows require
`metadata.anthropic_training_opt_in=true`; benchmark rows can never opt in.

## 3. Kaggle install and doctor

Run these in the cloned repository. The lock intentionally excludes Torch and
`--no-deps` prevents pip from replacing Kaggle's CUDA-matched installation.

```bash
python -m pip install --no-deps -r deploy/kaggle/requirements.lock
python -m pip install --no-deps -e .
python -m sycophancy_rl doctor
```

## 4. Exact first-experiment workflow

Run the 20-row BEFORE smoke first:

```bash
python -m sycophancy_rl benchmark --run-name qwen05b-before-smoke20 --model-id Qwen/Qwen2.5-0.5B-Instruct --benchmark-path data/anthropic_experiment/benchmark.jsonl --prompt-variants original --max-examples 20 --load-in-4bit --batch-size 1
```

Then run the full 3,017-row BEFORE benchmark:

```bash
python -m sycophancy_rl benchmark --run-name qwen05b-before-full --model-id Qwen/Qwen2.5-0.5B-Instruct --benchmark-path data/anthropic_experiment/benchmark.jsonl --prompt-variants original --load-in-4bit --batch-size 1
```

Run real online GRPO smoke training. It records global-step movement, one LoRA
parameter delta, trainable count, rewards/loss metrics, adapter save, saved
adapter reload, and a reload inference response; any missing optimizer update
fails the smoke.

```bash
python -m sycophancy_rl train --run-name qwen05b-grpo-smoke --profile kaggle_online_smoke --model-id Qwen/Qwen2.5-0.5B-Instruct --train-path data/anthropic_experiment/train.jsonl --validation-path data/anthropic_experiment/validation.jsonl --benchmark-path data/anthropic_experiment/benchmark.jsonl --rollout-mode online --max-pushback-turns 1
```

Run the full 24,134-row development experiment with 3,017-row validation:

```bash
python -m sycophancy_rl train --run-name qwen05b-grpo-full --profile qwen25_05b_online --model-id Qwen/Qwen2.5-0.5B-Instruct --train-path data/anthropic_experiment/train.jsonl --validation-path data/anthropic_experiment/validation.jsonl --benchmark-path data/anthropic_experiment/benchmark.jsonl --rollout-mode online --max-pushback-turns 1
```

Run AFTER on the same source IDs. Copy `benchmark_example_id_sha256` from the
full BEFORE manifest into the required argument:

```bash
python -m sycophancy_rl benchmark --run-name qwen05b-after-full --model-id Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/checkpoints/qwen05b-grpo-full/final_adapter --benchmark-path data/anthropic_experiment/benchmark.jsonl --prompt-variants original --expected-example-id-sha256 BEFORE_ID_SHA256 --load-in-4bit --batch-size 1
python -m sycophancy_rl compare --before outputs/evaluations/qwen05b-before-full/responses.jsonl --after outputs/evaluations/qwen05b-after-full/responses.jsonl
```

For subjective Anthropic rows, the primary outcome is an independent-choice
rate, not factual accuracy. Validation can guide training diagnostics and early
stopping. The held-out benchmark must not influence hyperparameters after the
experiment settings are selected.

## Outputs

A completed run stores the exact plan and environment, before/after raw
responses and summaries, adapter checkpoints, dataset/model hashes, failure or
completion status, and a paired comparison. Publish numbers only from these
artifacts; keep invalid and refused answers in the denominator.

If a saved run exposes a confirmed parser-contract defect, re-score its raw
responses without loading model weights or retraining:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl.evaluation.rescore_archive `
  --archive C:\path\to\experiment-artifacts.zip `
  --output outputs\rescored-experiment
```

This command reads only the BEFORE/AFTER JSONL members, preserves the source
archive, and marks every derived report as post-hoc. Do not present a re-scored
report as the original preregistered benchmark result.

## Development verification

```powershell
.\.venv\Scripts\ruff.exe check src tests deploy\kaggle\runner.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt
docker compose -f docker-compose.trainer.yml --profile qwen05b config --quiet
```

No automated test downloads model weights or datasets.

The CPU checks verify packaging, schemas, persistence, reward/parser behavior,
experiment planning, and mocked trainer integration. A real adapter save/reload,
before/after model inference, CUDA QLoRA, and GRPO optimization still require a
machine with the pinned training extras and suitable GPU; no GPU result is
claimed by this repository.

## Documentation

- `docs/README.md` - documentation index
- `docs/architecture.md` - package ownership and runtime flow
- `docs/experiment_protocol.md` - reproducible before/after contract
- `docs/data_governance.md` - allowed training and benchmark data
- `docs/reward_design.md` - reward components and failure modes
- `docs/limitations.md` - current research and operational limitations
- `docs/supported_models.md` - pins, capabilities, and custom-model rules
- `docs/local_runner.md` and `docs/kaggle_runner.md` - operator guides
- `docs/threat_model.md` - what the container does and does not isolate
- `docs/industrial_readiness.md` - verified and externally pending gates

## Release status

The implementation is a release candidate. The first Qwen2.5-0.5B Kaggle smoke
still requires an operator GPU/account and is not claimed as completed here. Larger
Qwen/Mistral 7B profiles remain available as optional workflows. The source code is
available under Apache-2.0; dataset and base-model licenses remain independent and
must be reviewed by each operator. See `LICENSE` and `docs/licensing.md`.
