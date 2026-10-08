#!/bin/bash
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then echo "usage: $0 <checkpoint.pth> [num_envs=16] [repeats=3]" >&2; exit 1; fi
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
if [ "${HEADLESS:-1}" = 0 ]; then . "$(dirname "$0")/gui_gpu_env.sh"; fi
. "$(dirname "$0")/runtime_env.sh"
args=()
if [ -n "${BOX_SIZE:-}" ]; then args+=(--push_box_size "$BOX_SIZE"); fi
if [ "${HEADLESS:-1}" != 0 ]; then args+=(--headless); fi
if [ -n "${EPISODE_LENGTH:-}" ]; then args+=(--episode_length "$EPISODE_LENGTH"); fi
if [ -n "${SEED:-}" ]; then args+=(--seed "$SEED"); fi
python tokenhsi/run.py --task HumanoidMAPushDoor \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_task_rsi.yaml \
 --motion_file tokenhsi/data/dataset_push_door_stage1.yaml \
 --checkpoint "$1" --num_agents 2 --num_envs "${2:-16}" --num_objects 4 \
 --eval_repeats "${3:-3}" --task_graph "${TASK_GRAPH:-random}" \
 --output_path "${OUTPUT_PATH:-output/push_door_stage1_task_rsi_eval}" --test --eval --no_video "${args[@]}"
