#!/bin/bash
# [num_agents=4] [num_envs=1024] [num_objects=12]
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
. "$(dirname "$0")/runtime_env.sh"
set -eu
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_shared_region_success train
set -- --num_agents "${1:-4}" --num_envs "${2:-1024}" --num_objects "${3:-12}" --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-output/stage1/Stage1V2Fixed_00060000.pth}
  if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "Stage1 checkpoint missing: $STAGE1_CHECKPOINT; set STAGE1_CHECKPOINT=/path/to/fixed.pth" >&2
    exit 1
  fi
  set -- "$@" --transfer_checkpoint "$STAGE1_CHECKPOINT"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
set -- "$@" --task_camera "${TASK_CAMERA:-stack}"
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
if [ -n "${MINIBATCH_SIZE:-}" ]; then set -- "$@" --minibatch_size "$MINIBATCH_SIZE"; fi
python tokenhsi/run.py --task HumanoidMASharedSupport \
  --cfg_env "${SHARED_SUPPORT_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_shared_region_success.yaml}" \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_shared_region_success.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --output_path "$OUTPUT_PATH" --experiment Stage2SharedRegionSuccess --seed "${SEED:-42}" --no_video "$@"
