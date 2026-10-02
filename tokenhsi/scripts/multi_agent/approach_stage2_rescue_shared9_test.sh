#!/bin/bash
# [STAGE1_ONLY=1] <checkpoint.pth> [num_agents=2] [num_envs=16] [num_objects=4] [eval_repeats=3]
# Default Shared9 evaluation preserves all nine pairs and supports objects >= humans >= 2.
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
EVAL_SAMPLER=${EVAL_SAMPLER:-shared9}
case "$EVAL_SAMPLER" in
  klclimb50|shared9)
    if [ "$NUM_AGENTS" -lt 2 ] || [ "$NUM_OBJECTS" -lt "$NUM_AGENTS" ]; then
      echo "Evaluation requires objects >= humans >= 2" >&2; exit 1
    fi ;;
  *) echo "EVAL_SAMPLER must be shared9 or klclimb50" >&2; exit 1 ;;
esac
if [ -n "${RELATION_GRAPH:-}${BOX_SIZE_RANGE:-}" ]; then echo "Shared9 uses typed assets; use TASK_GRAPH for pair selection" >&2; exit 1; fi
set -- --task_graph "${TASK_GRAPH:-random_scenario}" --task_camera "${TASK_CAMERA:-stack}" --eval_stage2_sampler "$EVAL_SAMPLER"
DEFAULT_OUTPUT=output/approach_stage2_rescue_shared9
case "${STAGE1_ONLY:-0}" in
  0) ;;
  1) set -- "$@" --stage1_only; DEFAULT_OUTPUT=$DEFAULT_OUTPUT/stage1_only ;;
  *) echo 'STAGE1_ONLY must be 0 or 1' >&2; exit 1 ;;
esac
if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_rescue_shared9.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_shared9.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
  --num_agents "$NUM_AGENTS" --num_envs "$NUM_ENVS" --num_objects "$NUM_OBJECTS" \
  --eval_repeats "$NUM_REPEATS" --eval_skills "${EVAL_SKILLS:-loco}" \
  --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
  --output_path "${OUTPUT_PATH:-$DEFAULT_OUTPUT}" \
  --no_video --test --eval "$@"
