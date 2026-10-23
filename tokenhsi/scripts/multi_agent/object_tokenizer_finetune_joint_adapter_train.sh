#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
export PYTHONDONTWRITEBYTECODE=1
. "$(dirname "$0")/runtime_env.sh"
set -- --mode train --variant joint_adapter --envs "${1:-2048}" --output "${OUTPUT_PATH:-output/object_tokenizer_finetune/joint_adapter/train}" --seed "${SEED:-42}" --headless
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max-iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT"; fi
if [ -n "${ASSETS_PER_CATEGORY:-}" ]; then set -- "$@" --assets-per-category "$ASSETS_PER_CATEGORY"; fi
exec python ports/ksh_object_finetune/launch.py "$@"
