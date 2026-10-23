#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
if [ "$TOKENHSI_GPU" != 6 ]; then echo "OPEN/PUSH uses GPU 6 only" >&2; exit 1; fi
# [num_agents=2] [num_envs=2048] [num_objects=4]
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-4}
set -- --task_graph random_open_push
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction.yaml \
  --motion_file tokenhsi/data/dataset_open_push.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction}" \
  --experiment OpenPushLateRSIHingeFriction --headless --no_video "$@"
