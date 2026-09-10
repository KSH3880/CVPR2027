#!/bin/bash
# [num_agents=2] [num_envs=2048] [num_objects=3]; SMOKE=1 uses small batches.
# RELATION_VARIANT=state2 selects delta weight 2 and a separate output directory.
# RELATION_VARIANT=state2_signed keeps state2 weights and selects signed linear progress.
. "$(dirname "$0")/runtime_env.sh"
set -eu
. "$(dirname "$0")/relation_variant.sh"
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-3}
TRAIN_CFG=tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml
if [ "${SMOKE:-0}" = 1 ]; then
    NUM_ENVS=${2:-32}
    TRAIN_CFG=tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_smoke.yaml
    MAX_ITERATIONS=${MAX_ITERATIONS:-3}
fi
set --
if [ -n "$RELATION_EXPERIMENT" ]; then
    set -- "$@" --experiment "$RELATION_EXPERIMENT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then
    set -- "$@" --max_iterations "$MAX_ITERATIONS"
fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
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
    --headless --no_video "$@"
