#!/bin/bash
# [num_agents=1] [num_envs=2048] [num_objects=1]
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-1}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-1}
set -- --task_graph open
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMADoorHold \
  --cfg_train tokenhsi/data/cfg/train/rlg/door_open_hold_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml \
  --motion_file tokenhsi/data/dataset_door_bones_left.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/door_open_hold_amp}" \
  --experiment DoorOpenHoldAMP --headless --no_video "$@"
