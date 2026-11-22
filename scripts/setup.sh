#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-python3}"
torch_channel="${TORCH_CHANNEL:-cpu}"

if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -c "import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))" || {
    echo "Set PYTHON_BIN to a Python 3.11 or 3.12 executable." >&2
    exit 1
  }
  "$python_bin" -m venv .venv
fi
.venv/bin/python -c "import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))" || {
  echo "Use a virtual environment created with Python 3.11 or 3.12." >&2
  exit 1
}

if [[ "$(uname -s)" == "Darwin" ]]; then
  .venv/bin/python -m pip install "torch==2.14.1"
else
  .venv/bin/python -m pip install "torch==2.14.1" \
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
