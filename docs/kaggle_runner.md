# Kaggle runner

The Kaggle workflow separates governed data from executable code. Local staging
creates two reviewable directories; it does not upload them or start a kernel.
The staged kernel uses the same frozen `ExperimentPlan` and `Pipeline` as the
local runner.

## Prerequisites

- A Kaggle account and API credentials configured by the operator.
- A Kaggle GPU quota suitable for a 7B 4-bit run.
- Internet enabled on the kernel for the first public-model download, or a
  separately mounted model source.
- Optional Kaggle secret `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) for gated/private
  Hub resources. The runner exports the secret but never prints its value.

## 1. Freeze the experiment plan

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

Do not pass `--execute` here. The command validates and freezes the plan without
loading a model.

## 2. Stage governed data

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl kaggle stage-data `
  --train data/generated/user/my-dataset/splits/train.jsonl `
  --validation data/generated/user/my-dataset/splits/validation.jsonl `
  --benchmark data/benchmarks/anthropic_sycophancy.jsonl `
  --dataset YOUR_KAGGLE_USER/sycophancy-rl-data `
  --output .kaggle-data-build
```

Staging rejects smoke fixtures, benchmark rows in training/validation, and ID
overlap. It copies available split/import/source manifests and writes hashes in
`data-manifest.json`.

Review, then explicitly upload with the Kaggle CLI:

```powershell
.\.venv\Scripts\python.exe -m kaggle datasets create -p .kaggle-data-build
```

Use `datasets version` instead of `datasets create` when updating an existing
Kaggle dataset. Keep the dataset private when it contains non-public data.

## 3. Stage the self-contained kernel

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl kaggle stage `
  --plan experiment-plan.json `
  --username YOUR_KAGGLE_USER `
  --dataset YOUR_KAGGLE_USER/sycophancy-rl-data `
  --output .kaggle-build
```

The directory contains the runner, frozen plan, pinned dependency lock, source
package, project metadata, and private GPU kernel metadata. Validate it:

```powershell
.\.venv\Scripts\python.exe -m sycophancy_rl kaggle validate `
  --metadata .kaggle-build/kernel-metadata.json
.\.venv\Scripts\python.exe -m sycophancy_rl kaggle push `
  --metadata .kaggle-build/kernel-metadata.json `
  --dry-run
```

## 4. Upload and run

After reviewing both staging directories:

```powershell
.\.venv\Scripts\python.exe -m kaggle kernels push -p .kaggle-build
```

This is the first command that uploads executable code. Kaggle then schedules
the private GPU kernel. `deploy/kaggle/runner.py` locates the mounted governed
dataset, installs only mismatched pinned packages, checks CUDA, runs
before/train/after/compare, and writes results under `/kaggle/working`.

## Outputs and resume

Download the complete `/kaggle/working/outputs` and
`/kaggle/working/checkpoints` trees. They contain manifests, raw responses,
summaries, comparison results, status, and adapter files. A SIGTERM records an
interrupted state. To restore an externally persisted checkpoint, mount it and
set `SYCO_RESUME_CHECKPOINT_ROOT` to its parent before rerunning the same frozen
identity.

Kaggle runtime storage is ephemeral. Do not consider a run complete until its
artifacts are downloaded or versioned into a private output dataset.

## Honest validation status

Unit tests cover staging, path replacement, governance rejection, and runner
dispatch without network access. A real Kaggle GPU execution is an external
release gate and must be recorded separately; static tests are not evidence
that a 7B job completed on Kaggle.
