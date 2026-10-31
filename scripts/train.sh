#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly after checking existing GPU workloads}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/output_etc/torch_extensions}"
export CUDA_VISIBLE_DEVICES="$TOKENHSI_GPU"
export PYTHONPATH="$PWD:$PWD/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
args=(--task HumanoidTokenHSIPushKD --cfg_env tokenhsi/data/cfg/multi_task/push_kd.yaml --cfg_train tokenhsi/data/cfg/train/rlg/push_kd.yaml --motion_file tokenhsi/data/dataset_push_kd.yaml --num_envs 2048 --headless --sim_device cuda:0 --rl_device cuda:0 --pipeline gpu --seed "${SEED:-42}" --output_path "${OUTPUT_PATH:-output/push_kd}")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then args+=(--resume 1 --checkpoint "$RESUME_CHECKPOINT"); fi
if [[ -n "${MAX_ITERATIONS:-}" ]]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
exec python tokenhsi/run.py "${args[@]}"
