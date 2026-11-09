#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CHECKPOINT:?Set student CHECKPOINT}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
if [[ "${EVAL_PROTOCOL:-cvpr2027_single}" == legacy ]]; then
  exec bash scripts/eval_legacy.sh
fi
if [[ "${EVAL_PROTOCOL:-cvpr2027_single}" != cvpr2027_single ]]; then
  echo 'Supported EVAL_PROTOCOL: cvpr2027_single, legacy' >&2; exit 2
fi
args=(--checkpoint "$CHECKPOINT" --gpu "$TOKENHSI_GPU")
if [[ -n "${EVAL_SEEDS:-}" ]]; then
  read -r -a seeds <<< "$EVAL_SEEDS"
  args+=(--seeds "${seeds[@]}")
fi
args+=(--protocol "${EVAL_REVISION:-1109}")
if [[ -n "${EVAL_TASK:-}" ]]; then args+=(--tasks "$EVAL_TASK"); fi
if [[ -n "${EVAL_OUTPUT:-}" ]]; then args+=(--output "$EVAL_OUTPUT"); fi
exec "${TOKENHSI_RUNTIME:-/home/injesus1010/anaconda3/envs/tokenhsi118}/bin/python" scripts/eval_cvpr2027.py "${args[@]}"
