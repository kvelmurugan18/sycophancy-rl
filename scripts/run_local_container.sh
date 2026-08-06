#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

model_size="${MODEL_SIZE:-7b}"
case "$model_size" in
  7b) service="trainer-7b" ;;
  smollm) service="trainer-smollm" ;;
  *) echo "MODEL_SIZE must be 7b or smollm" >&2; exit 2 ;;
esac

export SYCO_RUN_ID="${SYCO_RUN_ID:-${model_size}-local-seed42}"
: "${SYCO_TRAIN_PATH:?Set SYCO_TRAIN_PATH to a real container training split path}"
: "${SYCO_VALIDATION_PATH:?Set SYCO_VALIDATION_PATH to a real container validation split path}"
: "${SYCO_BENCHMARK_PATH:?Set SYCO_BENCHMARK_PATH to the container benchmark path}"
export SYCO_TRAIN_PATH SYCO_VALIDATION_PATH SYCO_BENCHMARK_PATH
export SYCO_MAX_EXAMPLES="${SYCO_MAX_EXAMPLES:-200}"
export SYCO_BATCH_SIZE="${SYCO_BATCH_SIZE:-1}"

docker compose -f docker-compose.trainer.yml --profile "$model_size" up \
  --abort-on-container-exit --exit-code-from "$service" "$service" "$@"
