#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
# [num_agents=2] [num_envs=2048] [num_objects=2]
if [ -z "${TOKENHSI_CONDA_ENV:-}" ]; then
  case "${CONDA_DEFAULT_ENV:-}" in
    ''|base) export TOKENHSI_CONDA_ENV=tokenhsi_koo ;;
    *) export TOKENHSI_CONDA_ENV=$CONDA_DEFAULT_ENV ;;
  esac
fi
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-2}
set -- --task_graph push
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/push_toy_random_at_10deg_at_state.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_state.yaml \
  --motion_file tokenhsi/data/dataset_open_push.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/push_toy_random_at_10deg_at_state}" \
  --experiment PushToyRandomAT10DegATState --headless --no_video "$@"
