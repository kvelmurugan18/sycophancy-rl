#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-python3}"
torch_channel="${TORCH_CHANNEL:-cpu}"

if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  .venv/bin/python -m pip install "torch==2.10.0"
else
  .venv/bin/python -m pip install "torch==2.10.0" \
    --index-url "https://download.pytorch.org/whl/${torch_channel}"
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ".[dev,kaggle]"
.venv/bin/python -m pip check

# Prepare data/processed/training_pool.jsonl and data/splits/ without
# overwriting existing artifacts. A fresh clone generates the smoke fixture
# pool; an existing repository with real data keeps the user's training pool
# and split files untouched.
.venv/bin/python -m sycophancy_rl.data_prep.setup_data ensure-fixtures
.venv/bin/python -m sycophancy_rl.data_prep.setup_data ensure-splits
.venv/bin/python -m pytest
.venv/bin/python -m sycophancy_rl.training.train_grpo \
  --profile smoke --allow-cpu --no-4bit --preflight-only
