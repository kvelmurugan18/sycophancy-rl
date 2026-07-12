#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launches the GRPO training pipeline for the Sycophancy RL project.
#
# IMPORTANT: This script must be run from the root of the project (the
# directory that contains `src/`, `data/`, `server/`, etc.). Running it
# from anywhere else will break the relative imports inside
# `src/training/train_grpo.py` (which uses `from src.reward.reward_fn
# import composite_reward_func` and the unified dataset path
# `data/processed/merged_episodes.jsonl`).
# -----------------------------------------------------------------------------

set -e

# Make `from src...` imports work when the script is invoked as a bare
# shell script (rather than as `python -m ...`, which puts the CWD on
# `sys.path` automatically). The leading `.:` covers the project root;
# `$PYTHONPATH` is preserved so any user-level entries (e.g. site-packages
# overrides) still apply.
export PYTHONPATH=.:${PYTHONPATH:-}

# NOTE: If you have multiple GPUs and want to scale the GRPO run across
# them, swap `python` for one of the following launchers — the training
# script itself does not need to change:
#   - `accelerate launch --num_processes <N> -m src.training.train_grpo`
#   - `torchrun --nproc_per_node <N> -m src.training.train_grpo`
# The single-process default below is the right call for a 16 GB
# consumer-GPU budget; multi-GPU is an explicit opt-in because GRPO's
# advantage estimation requires careful coordination between processes.
python -m src.training.train_grpo
