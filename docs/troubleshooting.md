# Troubleshooting

## `PyTorch is not installed`

Install the training dependencies with the setup script or the `train` extra.
Use the PyTorch channel compatible with the host driver.

## `Profile 'qlora_7b_16gb' needs at least 14 GiB VRAM`

The preflight intentionally blocks the 7B profile. Use a suitable GPU, close
other GPU processes, or select the smaller registered SmolLM profile. Do not
bypass the check and present the result as a supported configuration.

## CUDA out of memory after preflight

Keep 4-bit loading enabled, benchmark batch size at `1`, and reduce completion
length or concurrent work. Restarting the container can clear fragmentation.
The pipeline releases benchmark model memory before training, but drivers and
other processes still consume VRAM.

## `Training pool is missing`

For offline tests run `python -m sycophancy_rl.data_prep.setup_data ensure-fixtures`. For
a real run, import a user dataset and point `--train-path` and
`--validation-path` to its generated splits.

## `real profiles cannot train on smoke fixtures`

The committed 12-row fixture is test-only. Import or prepare a non-fixture
dataset; do not remove the guard.

## `benchmark rows cannot be used for training`

The Anthropic dataset is evaluation-only. Use a separately licensed training
source and keep the benchmark path only in `--benchmark-path`.

## `Imported dataset already exists`

Imports are immutable. Choose a new `--dataset-name` for the revised source
instead of overwriting the previous provenance.

## Docker cannot access the GPU

Verify the NVIDIA driver, Docker Desktop GPU support, and NVIDIA container
runtime outside this project. Static `docker compose ... config` success proves
only YAML rendering, not device access.

## Hugging Face model is unavailable in offline mode

Set the three offline flags to `0` for the first authorized download. After the
model and tokenizer are fully cached in `hf-cache`, set them to `1` for repeat
runs. Never commit an access token to `.env` or source files.

## Kaggle cannot find the mounted dataset

The dataset must contain `splits/train.jsonl`, `splits/validation.jsonl`, and
`benchmarks/anthropic_sycophancy.jsonl`. Confirm the staged dataset slug matches
the kernel metadata and that Kaggle mounted the expected version.

## Illegal manifest transition or incompatible resume

Do not edit manifests by hand. Use a new run ID, or resume only the identical
created/failed/interrupted run with unchanged model, data, seed, and reward
identity. Preserve the old run for diagnosis.
