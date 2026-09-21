#!/bin/bash
# <checkpoint> [num_agents=2] [num_envs=16] [num_objects=3] [eval_repeats=3]
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]" >&2
    exit 1
fi
CKPT=$1
NUM_AGENTS=${2:-2}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-3}
NUM_REPEATS=${5:-3}
RELATION_ENV_CFG=tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1.yaml
RELATION_OUTPUT_DEFAULT=output/approach_distance_edge_context_stage1
set -- --task_graph "${TASK_GRAPH:-holding_sit}" --task_camera "${TASK_CAMERA:-stack}"
if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml \
    --cfg_env "$RELATION_ENV_CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --checkpoint "$CKPT" \
    --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
    --eval_repeats "$NUM_REPEATS" \
    --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
    --output_path "${OUTPUT_PATH:-$RELATION_OUTPUT_DEFAULT}" --no_video --test --eval "$@"
