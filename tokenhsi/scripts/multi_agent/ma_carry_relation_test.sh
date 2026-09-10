#!/bin/bash
# <checkpoint> [num_agents=2] [num_envs=16] [num_objects=3] [eval_repeats=3]
# Select the checkpoint's RELATION_VARIANT (v0, state2, state2_signed); metadata is strict.
. "$(dirname "$0")/runtime_env.sh"
set -eu
. "$(dirname "$0")/relation_variant.sh"
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <checkpoint.pth> [num_agents] [num_envs] [num_objects] [eval_repeats]" >&2
    exit 1
fi
CKPT=$1
NUM_AGENTS=${2:-2}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-3}
NUM_REPEATS=${5:-3}
set --
if [ -n "${SEED:-}" ]; then
    set -- "$@" --seed "$SEED"
fi
if [ "${HEADLESS:-1}" != 0 ]; then
    set -- "$@" --headless
fi
python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml \
    --cfg_env "$RELATION_ENV_CFG" \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint "$CKPT" \
    --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
    --eval_repeats "$NUM_REPEATS" \
    --eval_skills loco,pickUp,carryWith,putDown --eval_skill_probs 0.5,0.1,0.3,0.1 \
    --output_path "${OUTPUT_PATH:-$RELATION_OUTPUT_DEFAULT}" --no_video --test --eval "$@"
