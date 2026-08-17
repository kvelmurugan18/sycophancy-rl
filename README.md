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
- Explicit `online` (research default) and `prepared` (teacher-forced smoke) modes.
- Registered, commit-pinned SmolLM2 1.7B, Qwen2.5 7B, and Mistral 7B profiles.
- A `qlora_7b_16gb` profile for 4-bit NF4 LoRA + GRPO on a suitable GPU.
- CSV/JSON/JSONL user-dataset importer with immutable splits and provenance.
- Anthropic model-written sycophancy data as evaluation-only benchmark data.
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
- A GPU with at least 14 GiB available VRAM for the provided 7B QLoRA profile.
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
.\scripts\setup.ps1 -TorchChannel cu126
```

Linux:

```bash
TORCH_CHANNEL=cu126 bash scripts/setup.sh
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

## 2. Prepare the Anthropic evaluation benchmark

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl prepare-data --help
.\.venv\Scripts\python.exe -m sycophancy_rl.data_prep.prepare_anthropic_benchmark
```

This networked preparation step pins the source revision and writes a manifest.
The Anthropic rows are marked `benchmark_only`; the training loader rejects
them. Never use this benchmark as the training dataset.

## 3. Preview the 7B experiment

Planning performs validation and writes no weights:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl run `
  --run-id qwen25-7b-seed42 `
  --runner local `
  --profile qlora_7b_16gb `
  --model-id Qwen/Qwen2.5-7B-Instruct `
  --train-path data/generated/user/my-dataset/splits/train.jsonl `
  --validation-path data/generated/user/my-dataset/splits/validation.jsonl `
  --benchmark-path data/benchmarks/anthropic_sycophancy.jsonl `
  --batch-size 1
```

Omitting `--execute` is intentional: it returns the frozen plan and checks
without training.

## 4A. Run locally in the isolated container

Copy `.env.example` to `.env`, adjust only non-secret settings, and never
commit `.env`. Then run the complete workflow:

```powershell
.\scripts\run_local_container.ps1 `
  -ModelSize 7b `
  -RunId qwen25-7b-seed42 `
  -TrainPath /data/generated/user/my-dataset/splits/train.jsonl `
  -ValidationPath /data/generated/user/my-dataset/splits/validation.jsonl `
  -BenchmarkPath /data/benchmarks/anthropic_sycophancy.jsonl `
  -Build
```

Linux uses `bash scripts/run_local_container.sh` and the corresponding
`SYCO_*` environment variables. Details and security boundaries are in
`docs/local_runner.md`.

The diagnostic API uses `SYCO_DATASET_PATH`, `SYCO_ORIGINS` (or the equivalent
`SYCO_ALLOWED_ORIGINS` alias), `SYCO_CORS_ALLOW_CREDENTIALS`, `SYCO_HOST`, and
`SYCO_PORT`. Session state is stored in `data/sessions.sqlite3` by default;
set `SYCO_SESSION_DB` to another writable SQLite path. Every reset is persisted
immediately and every step atomically reloads, appends, and saves the complete
episode, so a new API process can continue the same session ID. External rollout
clients should send `/step.finish_reason` as
`"eos"` or `"length"`; omitting it keeps older clients working but disables
truncation detection for that request.

## 4B. Stage the same experiment for Kaggle

Create the frozen Kaggle plan:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl run `
  --run-id qwen25-7b-kaggle-seed42 `
  --runner kaggle `
  --profile qlora_7b_16gb `
  --model-id Qwen/Qwen2.5-7B-Instruct `
  --train-path data/generated/user/my-dataset/splits/train.jsonl `
  --validation-path data/generated/user/my-dataset/splits/validation.jsonl `
  --benchmark-path data/benchmarks/anthropic_sycophancy.jsonl `
  --batch-size 1 `
  --plan-output experiment-plan.json
```

Then stage the governed data and self-contained kernel:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl kaggle stage-data `
  --train data/generated/user/my-dataset/splits/train.jsonl `
  --validation data/generated/user/my-dataset/splits/validation.jsonl `
  --benchmark data/benchmarks/anthropic_sycophancy.jsonl `
  --dataset YOUR_KAGGLE_USER/sycophancy-rl-data

.\.venv\Scripts\python.exe -m sycophancy_rl kaggle stage `
  --plan experiment-plan.json `
  --username YOUR_KAGGLE_USER `
  --dataset YOUR_KAGGLE_USER/sycophancy-rl-data
```

Review the staged directories before uploading. `stage` never uploads or starts
training. See `docs/kaggle_runner.md` for the explicit upload/run procedure.

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
docker compose -f docker-compose.trainer.yml --profile 7b config --quiet
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

The implementation is a release candidate. A real Qwen/Mistral 7B GPU run and
a real Kaggle run still require operator hardware/accounts and have not been
claimed here. The source code is available under Apache-2.0; dataset and base
model licenses remain independent and must be reviewed by each operator. See
`LICENSE` and `docs/licensing.md`.
