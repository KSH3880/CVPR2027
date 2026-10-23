#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
# [num_agents=2] [num_envs=2048] [num_objects=4]
if [ -n "${TRANSFER_CHECKPOINT:-}" ]; then
  echo "Use the migrated full-state checkpoint through RESUME_CHECKPOINT; unset TRANSFER_CHECKPOINT." >&2
  exit 2
fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-4}
set -- --task_graph push
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5.yaml \
  --motion_file tokenhsi/data/dataset_push_only_push180.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/push_fixed130_fixed_face_midpoint_goal3to5}" \
  --experiment PushFixed130FixedFaceMidpointGoal3To5 --headless --no_video "$@"
