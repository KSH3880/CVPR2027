#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
export PYTHONDONTWRITEBYTECODE=1
. "$(dirname "$0")/runtime_env.sh"
if [ $# -lt 1 ]; then echo 'usage: <fine-tuned checkpoint> [envs=16] [repeats=3]' >&2; exit 1; fi
ckpt=$(realpath -e "$1")
set -- --mode test --variant carry_chair --checkpoint "$ckpt" --envs "${2:-16}" --repeats "${3:-3}" --output "${OUTPUT_PATH:-output/object_tokenizer_finetune/carry_chair/test}" --seed "${SEED:-42}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${ASSETS_PER_CATEGORY:-}" ]; then set -- "$@" --assets-per-category "$ASSETS_PER_CATEGORY"; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode-length "$EPISODE_LENGTH"; fi
exec python ports/ksh_object_finetune/launch.py "$@"
