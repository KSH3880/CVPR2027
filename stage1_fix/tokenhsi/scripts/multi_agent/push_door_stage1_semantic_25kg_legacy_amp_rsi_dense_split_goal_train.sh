#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
export SEED=${SEED:-8427}
args=()
if [ -n "${BOX_SIZE:-}" ]; then args+=(--push_box_size "$BOX_SIZE"); fi
if [ -n "${MAX_ITERATIONS:-}" ]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then echo "This comparison starts from scratch; unset RESUME_CHECKPOINT." >&2; exit 1; fi
if [ -n "${SEED:-}" ]; then args+=(--seed "$SEED"); fi
python tokenhsi/run.py --task HumanoidMAPushDoor \
 --cfg_env "${CFG_ENV:-tokenhsi/data/cfg/multi_agent/push_door_stage1_semantic_25kg_legacy_amp_rsi_dense_split_goal.yaml}" \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_push_door_stage1_semantic_25kg_legacy_amp_rsi_dense_split_goal.yaml \
 --motion_file tokenhsi/data/dataset_push_door_stage1.yaml \
 --num_agents 2 --num_envs "${NUM_ENVS:-2048}" --num_objects 4 \
 --output_path "${OUTPUT_PATH:-output/push_door_stage1_semantic_25kg_legacy_amp_rsi_dense_split_goal}" --experiment PushDoor25kgDoorGoalSplit \
 --headless --no_video "${args[@]}"
