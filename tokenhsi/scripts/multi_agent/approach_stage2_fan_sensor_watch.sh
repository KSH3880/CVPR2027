#!/bin/bash
# STAGE1_CHECKPOINT=<source.pth> [num_agents=2] [num_envs=4] [num_objects=3]
export SENSOR_VIS=${SENSOR_VIS:-1}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_sm120}
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-4}
NUM_OBJECTS=${3:-3}
if [ "$NUM_AGENTS" != 2 ] || [ "$NUM_OBJECTS" != 3 ]; then
  echo 'Fan watch requires 2 agents and 3 objects.' >&2
  exit 1
fi
case "$NUM_ENVS" in
  ''|*[!0-9]*) echo 'num_envs must be a positive integer.' >&2; exit 1 ;;
esac
if [ "$NUM_ENVS" -lt 1 ]; then echo 'num_envs must be at least 1.' >&2; exit 1; fi
set -- --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  if [ -z "${STAGE1_CHECKPOINT:-}" ] || [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo 'Set STAGE1_CHECKPOINT to an existing Stage-1 schema-9 .pth' >&2
    exit 1
  fi
  set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMACliff \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_cliff_sensor_watch.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_fan_sensor.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/approach_stage2_fan_sensor_576_100_10m_watch}" \
  --experiment ApproachStage2FanSensorWatch --no_video "$@"
