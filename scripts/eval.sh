#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
: "${CHECKPOINT:?Set student CHECKPOINT}"
if [[ "${EVAL_TASK:-push}" == traj ]]; then echo "TRAJ is excluded" >&2; exit 2; fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/output_etc/torch_extensions}"
export CUDA_VISIBLE_DEVICES="$TOKENHSI_GPU"
export PYTHONPATH="$PWD:$PWD/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
exec python tokenhsi/run.py --task HumanoidTokenHSIPushKD --cfg_env tokenhsi/data/cfg/multi_task/push_kd.yaml --cfg_train tokenhsi/data/cfg/train/rlg/push_kd.yaml --motion_file tokenhsi/data/dataset_push_kd.yaml --test --eval --eval_task "${EVAL_TASK:-push}" --checkpoint "$CHECKPOINT" --num_envs "${NUM_ENVS:-512}" --headless --sim_device cuda:0 --rl_device cuda:0 --pipeline gpu --seed "${SEED:-0}"
