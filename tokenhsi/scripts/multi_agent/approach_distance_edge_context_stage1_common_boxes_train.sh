#!/bin/bash
# [num_agents=2] [num_envs=2048] [num_objects=3]; Stage-1 common-box scratch training.
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-3}
RELATION_ENV_CFG=tokenhsi/data/cfg/multi_agent/approach_distance_edge_context_stage1_common_boxes.yaml
RELATION_OUTPUT_DEFAULT=output/approach_distance_edge_context_stage1_common_boxes
TRAIN_CFG=tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml
set -- --task_graph random_stage1
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train "$TRAIN_CFG" --cfg_env "$RELATION_ENV_CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
    --output_path "${OUTPUT_PATH:-$RELATION_OUTPUT_DEFAULT}" \
    --experiment ApproachDistanceEdgeContextStage1CommonBoxes \
    --headless --no_video "$@"
