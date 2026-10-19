#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
lift_gpu_index=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
[ "$lift_gpu_index" = 6 ] || { echo 'Lift uses physical GPU 6 only' >&2; exit 1; }
. "$SCRIPT_DIR/runtime_env.sh"
args=()
if [ -n "${MAX_ITERATIONS:-}" ]; then args+=(--max_iterations "$MAX_ITERATIONS"); fi
if [ -n "${SEED:-}" ]; then args+=(--seed "$SEED"); fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--checkpoint "$RESUME_CHECKPOINT" --resume 1); fi
python tokenhsi/run.py --task HumanoidMALift \
 --cfg_env tokenhsi/data/cfg/multi_agent/lift_stage1_teamhoi.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_lift_stage1_teamhoi.yaml \
 --motion_file tokenhsi/data/dataset_lift_stage1.yaml \
 --num_agents 2 --num_objects 2 --num_envs 2048 \
 --output_path "${OUTPUT_PATH:-output/lift_stage1_teamhoi}" --experiment LiftStage1TeamHOI --headless --no_video "${args[@]}"
