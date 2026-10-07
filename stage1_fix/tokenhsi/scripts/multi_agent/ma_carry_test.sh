#!/bin/bash
# Multi-agent carry test / visualisation.
#   sh tokenhsi/scripts/multi_agent/ma_carry_test.sh <checkpoint.pth> [num_agents] [num_envs] [num_objects] [num_repeats] [eval_skills] [eval_skill_probs]
# Set HEADLESS=0 and run through the VNC helper to see the scene.

. "$(dirname "$0")/runtime_env.sh"

if [ $# -lt 1 ]; then
    echo "usage: $0 <checkpoint.pth> [num_agents] [num_envs] [num_objects] [num_repeats] [eval_skills] [eval_skill_probs]" >&2
    exit 1
fi

CKPT=$1
NUM_AGENTS=${2:-1}
NUM_ENVS=${3:-16}
NUM_OBJECTS=${4:-0}
NUM_REPEATS=${5:-3}
EVAL_SKILLS=${6:-loco,pickUp,carryWith,putDown}
EVAL_SKILL_PROBS=${7:-0.5,0.1,0.3,0.1}
# EVAL_SKILLS=${6:-carryWith,putDown}
# EVAL_SKILL_PROBS=${7:-0.5,0.5}
HEADLESS_FLAG=""
if [ "${HEADLESS:-1}" != "0" ]; then
    HEADLESS_FLAG="--headless"
fi

if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi

python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml \
    --cfg_env tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint "$CKPT" \
    --num_envs "$NUM_ENVS" \
    --num_agents "$NUM_AGENTS" \
    --num_objects "$NUM_OBJECTS" \
    --eval_repeats "$NUM_REPEATS" \
    --eval_skills "$EVAL_SKILLS" \
    --eval_skill_probs "$EVAL_SKILL_PROBS" \
    --no_video \
    --output_path output/ma_carry \
    --test \
    --eval ${HEADLESS_FLAG}
