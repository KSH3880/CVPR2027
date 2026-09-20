#!/bin/bash
# [num_agents=2] [num_envs=2048] [num_objects=3]; PRO 6000 training baseline.
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-3}
RELATION_ENV_CFG=tokenhsi/data/cfg/multi_agent/approach_distance_success_ontop_mixed.yaml
RELATION_OUTPUT_DEFAULT=output/approach_distance_success_ontop_mixed
TRAIN_CFG=tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml
set --
if [ -n "${MAX_ITERATIONS:-}" ]; then
    set -- "$@" --max_iterations "$MAX_ITERATIONS"
fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
fi
if [ -n "${TRANSFER_CHECKPOINT:-}" ]; then
    if [ -n "${RESUME_CHECKPOINT:-}" ]; then
        echo "Choose TRANSFER_CHECKPOINT for new training or RESUME_CHECKPOINT for resume." >&2
        exit 1
    fi
    set -- "$@" --transfer_checkpoint "$TRANSFER_CHECKPOINT"
fi
if [ -n "${SEED:-}" ]; then
    set -- "$@" --seed "$SEED"
fi
python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$RELATION_ENV_CFG" \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
    --output_path "${OUTPUT_PATH:-$RELATION_OUTPUT_DEFAULT}" \
    --experiment ApproachDistanceSuccessOntopMixed \
    --headless --no_video "$@"
