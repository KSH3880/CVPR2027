#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
# <checkpoint> [num_agents=2] [num_envs=16] [num_objects=2] [eval_repeats=3]
if [ -z "${TOKENHSI_CONDA_ENV:-}" ]; then
  case "${CONDA_DEFAULT_ENV:-}" in
    ''|base) export TOKENHSI_CONDA_ENV=tokenhsi_koo ;;
    *) export TOKENHSI_CONDA_ENV=$CONDA_DEFAULT_ENV ;;
  esac
fi
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo "usage: $0 <checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]" >&2
  exit 1
fi
CKPT=$(realpath "$1")
. "$(dirname "$0")/runtime_env.sh"
NUM_AGENTS=${2:-2}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-2}
NUM_REPEATS=${5:-3}
set -- --task_graph "push" --task_camera "${TASK_CAMERA:-stack}"
if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python ./tokenhsi/run.py --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/push_toy_random_at_10deg_at_state.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/push_toy_random_at_10deg_at_state.yaml \
  --motion_file tokenhsi/data/dataset_open_push.yaml --checkpoint "$CKPT" \
  --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" \
  --eval_repeats "$NUM_REPEATS" --eval_skills "${EVAL_SKILLS:-loco}" \
  --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "${OUTPUT_PATH:-output/push_toy_random_at_10deg_at_state}" \
  --no_video --test --eval "$@"
