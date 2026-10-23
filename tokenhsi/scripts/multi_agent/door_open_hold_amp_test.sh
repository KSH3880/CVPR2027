#!/bin/bash
# <checkpoint> [num_agents=1] [num_envs=16] [num_objects=1] [eval_repeats=3]
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo "usage: $0 <checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]" >&2
  exit 1
fi
CKPT=$1
NUM_AGENTS=${2:-1}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-1}
NUM_REPEATS=${5:-3}
set -- --task_graph "${TASK_GRAPH:-open}" --task_camera "${TASK_CAMERA:-stack}"
if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python ./tokenhsi/run.py --task HumanoidMADoorHold \
  --cfg_train tokenhsi/data/cfg/train/rlg/door_open_hold_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml \
  --motion_file tokenhsi/data/dataset_door_bones_left.yaml --checkpoint "$CKPT" \
  --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" \
  --eval_repeats "$NUM_REPEATS" --eval_skills "${EVAL_SKILLS:-loco}" \
  --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "${OUTPUT_PATH:-output/door_open_hold_amp}" \
  --no_video --test --eval "$@"
