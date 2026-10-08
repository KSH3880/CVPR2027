#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/gui_gpu_env.sh"
. "$(dirname "$0")/runtime_env.sh"
args=()
if [ -n "${BOX_SIZE:-}" ]; then args+=(--push_box_size "$BOX_SIZE"); fi
python tokenhsi/scripts/multi_agent/task_rsi_preview.py --task HumanoidMAPushDoor \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_task_rsi.yaml \
 --motion_file tokenhsi/data/dataset_push_door_stage1.yaml \
 --num_agents 2 --num_envs 1 --num_objects 4 --task_graph push_door \
 --output_path output/task_rsi_preview --test --no_video "${args[@]}"
