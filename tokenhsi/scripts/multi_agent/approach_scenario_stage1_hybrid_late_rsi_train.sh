#!/bin/bash
# [num_agents=2] [num_envs=2048] [num_objects=4]
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-4}
set -- --task_graph random_scenario
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${RESUME_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_shared_edge_encoder.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_hybrid_late_rsi.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/approach_scenario_stage1_hybrid_late_rsi}" \
  --experiment ApproachScenarioStage1HybridLateRsi --headless --no_video "$@"
