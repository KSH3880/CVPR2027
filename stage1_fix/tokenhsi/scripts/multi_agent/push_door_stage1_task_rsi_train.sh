#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
args=()
if [ -n "${BOX_SIZE:-}" ]; then args+=(--push_box_size "$BOX_SIZE"); fi
if [ -n "${MAX_ITERATIONS:-}" ]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--checkpoint "$RESUME_CHECKPOINT" --resume 1); fi
if [ -n "${SEED:-}" ]; then args+=(--seed "$SEED"); fi
python tokenhsi/run.py --task HumanoidMAPushDoor \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_door_stage1_task_rsi.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_task_rsi.yaml \
 --motion_file tokenhsi/data/dataset_push_door_stage1.yaml \
 --num_agents 2 --num_envs "${NUM_ENVS:-2048}" --num_objects 4 \
 --output_path "${OUTPUT_PATH:-output/push_door_stage1_task_rsi}" --experiment PushDoorStage1TaskRSI \
 --headless --no_video "${args[@]}"
