#!/bin/bash
# <stage2_at20_4agent_checkpoint.pth> [num_agents=4] [num_envs=16] [num_objects=8] [eval_repeats=3]
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo "usage: $0 <stage2_at20_4agent_checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]" >&2
  exit 1
fi
CKPT=$1
NUM_AGENTS=${2:-4}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-8}
NUM_REPEATS=${5:-3}
set -- --task_graph "${TASK_GRAPH:-random_scenario}" --task_camera "${TASK_CAMERA:-stack}"
if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
if [ -n "${RELATION_GRAPH:-}" ]; then set -- "$@" --relation_graph "$RELATION_GRAPH"; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/scripts/prepare_joint_carry.py
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_at20_4agent.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage2_at20_4agent.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
  --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" \
  --eval_repeats "$NUM_REPEATS" --eval_skills "${EVAL_SKILLS:-loco}" \
  --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "${OUTPUT_PATH:-output/stage2_at20_4agent}" \
  --no_video --test --eval "$@"
