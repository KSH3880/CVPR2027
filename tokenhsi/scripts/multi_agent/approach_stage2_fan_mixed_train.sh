#!/bin/bash
# [STAGE1_CHECKPOINT=output/stage1/8000.pth] [num_agents=2] [num_envs=2048] [num_objects=3]
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_sm120}
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-3}
set -- --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-output/stage1/8000.pth}
  if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo 'Set STAGE1_CHECKPOINT to an existing Stage-1 schema-9 .pth' >&2
    exit 1
  fi
  set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMAMixedSensor \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_mixed.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/approach_stage2_fan_mixed_cpa_static_wall}" \
  --experiment ApproachStage2FanMixed --headless --no_video "$@"
