#!/bin/bash
# Original warmstart checkpoint; evaluation-only goal override, no training.
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "usage: $0 <warmstart_checkpoint.pth> [agents=2] [envs=1] [objects=4] [repeats=100]" >&2
  exit 1
fi
set -- "$1" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-100}"
CKPT=$1
NUM_AGENTS=$2
NUM_ENVS=$3
NUM_OBJECTS=$4
NUM_REPEATS=$5
set -- --task_graph push --task_camera "${TASK_CAMERA:-stack}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}" python tokenhsi/scripts/multi_agent/view_open_push_random_goal.py \
  --task HumanoidMAOpenPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml \
  --motion_file tokenhsi/data/dataset_open_push_push180.yaml --checkpoint "$CKPT" \
  --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" \
  --eval_repeats "$NUM_REPEATS" --eval_skills loco --eval_skill_probs 1.0 \
  --seed "${SEED:-42}" --output_path "${OUTPUT_PATH:-output/open_push_zero_shot_random_goal}" \
  --test --eval --no_video "$@"
