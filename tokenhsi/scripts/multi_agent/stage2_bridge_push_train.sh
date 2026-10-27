#!/bin/bash
# [num_agents=2] [num_envs=2048] [num_objects=4]
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
. "$(dirname "$0")/runtime_env.sh"
set -eu
. "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/output_paths.sh"
tokenhsi_select_output stage2_bridge_push train
set -- --num_agents "${1:-2}" --num_envs "${2:-2048}" --num_objects "${3:-4}" --task_graph random_scenario
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  set -- "$@" --checkpoint "$RESUME_CHECKPOINT" --resume 1
else
  set -- "$@" --transfer_checkpoint "${STAGE1_CHECKPOINT:-../TokenHSI_stage1_push/output/stage1_v2_fixed_push_360_raw_full_plate_mps_resume_20261021_162424/Stage1V2FixedPush360Raw_21-16-24-52/nn/Stage1V2FixedPush360Raw_00013500.pth}"
fi
if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
python tokenhsi/run.py --task HumanoidMABridgePush \
  --cfg_env "${BRIDGE_PUSH_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_bridge_push.yaml}" \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_bridge_push.yaml \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml \
  --output_path "$OUTPUT_PATH" --experiment Stage2BridgePush --seed "${SEED:-42}" --headless --no_video "$@"
