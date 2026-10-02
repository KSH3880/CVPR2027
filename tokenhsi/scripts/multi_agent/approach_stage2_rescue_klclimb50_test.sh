#!/bin/bash
# [STAGE1_ONLY=1] <checkpoint.pth> [num_agents=2] [num_envs=16] [num_objects=4] [eval_repeats=3]
# Evaluation expands automatically for num_objects >= num_agents >= 2.
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo "usage: [STAGE1_ONLY=1] $0 <checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]" >&2
  exit 1
fi
CKPT=$1
NUM_AGENTS=${2:-2}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-4}
NUM_REPEATS=${5:-3}
set -- --task_graph "${TASK_GRAPH:-place_climb}" --task_camera "${TASK_CAMERA:-stack}"
DEFAULT_OUTPUT=output/approach_stage2_rescue_klclimb50
case "${STAGE1_ONLY:-0}" in
  0) ;;
  1) set -- "$@" --stage1_only; DEFAULT_OUTPUT=$DEFAULT_OUTPUT/stage1_only ;;
  *) echo 'STAGE1_ONLY must be 0 or 1' >&2; exit 1 ;;
esac
if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
if [ -n "${RELATION_GRAPH:-}" ]; then set -- "$@" --relation_graph "$RELATION_GRAPH"; fi
if [ -n "${BOX_SIZE_RANGE:-}" ]; then set -- "$@" --eval_box_size_range "$BOX_SIZE_RANGE"; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_rescue_klclimb50.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_klclimb50.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
  --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" \
  --eval_repeats "$NUM_REPEATS" --eval_skills "${EVAL_SKILLS:-loco}" \
  --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "${OUTPUT_PATH:-$DEFAULT_OUTPUT}" \
  --no_video --test --eval "$@"
