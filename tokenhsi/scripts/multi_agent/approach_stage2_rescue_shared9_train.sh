#!/bin/bash
# STAGE1_CHECKPOINT=<rescue.pth> [num_agents=2] [num_envs=2048] [num_objects=4]
. "$(dirname "$0")/runtime_env.sh"
set -eu
NUM_AGENTS=${1:-2}
NUM_ENVS=${2:-2048}
NUM_OBJECTS=${3:-4}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$TOKENHSI_ROOT/stage1/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth}
set -- --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  if [ -z "${STAGE1_CHECKPOINT:-}" ] || [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo 'Set STAGE1_CHECKPOINT to a rescue Stage-1 .pth' >&2
    exit 1
  fi
  set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
python ./tokenhsi/run.py --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_rescue_shared9.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_shared9.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs "$NUM_ENVS" --num_agents "$NUM_AGENTS" --num_objects "$NUM_OBJECTS" \
  --output_path "${OUTPUT_PATH:-output/approach_stage2_rescue_shared9}" \
  --experiment ApproachStage2RescueShared9 --headless --no_video "$@"
