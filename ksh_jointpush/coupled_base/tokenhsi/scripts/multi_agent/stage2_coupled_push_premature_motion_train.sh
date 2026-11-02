#!/usr/bin/env bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT=$(cd "$SCRIPT_DIR/../../.." && pwd)
export TOKENHSI_RUN_ENTRY=${TOKENHSI_RUN_ENTRY:-$PROJECT/../scripts/coupled_base_run.py}
export TORCH_EXTENSIONS_DIR="$PROJECT/output_etc/check/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1
export ENV_CONFIG=${ENV_CONFIG:-$PROJECT/tokenhsi/data/cfg/multi_agent/stage2_coupled_push_premature_motion.yaml}
export STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-$PROJECT/checkpoints/Stage1JointPushDensity3Cone90Finetune_00026000.pth}
if [ "${CHECK:-0}" = 1 ] || [ -n "${MAX_ITERATIONS:-}" ]; then
  export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT/output_etc/check/premature_motion_$(date +%Y%m%d_%H%M%S)}
else
  export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT/output/stage2_coupled_push_premature_motion_$(date +%Y%m%d_%H%M%S)}
fi
exec bash "$SCRIPT_DIR/launch.sh" stage2_coupled_push_region_goals train "$@"
