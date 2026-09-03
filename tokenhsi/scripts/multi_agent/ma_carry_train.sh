#!/bin/bash
# Multi-agent carry training.
#   sh tokenhsi/scripts/multi_agent/ma_carry_train.sh [num_agents] [num_envs] [num_objects]
# Defaults are sized for a 16GB GPU.

. "$(dirname "$0")/runtime_env.sh"

NUM_AGENTS=${1:-1}
NUM_ENVS=${2:-1024}
NUM_OBJECTS=${3:-0}
if [ -n "${MAX_ITERATIONS:-}" ]; then
    set -- --max_iterations "$MAX_ITERATIONS"
else
    set --
fi

python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry.yaml \
    --cfg_env tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --num_envs ${NUM_ENVS} \
    --num_agents ${NUM_AGENTS} \
    --num_objects ${NUM_OBJECTS} \
    --output_path output/ma_carry \
    --headless "$@"
