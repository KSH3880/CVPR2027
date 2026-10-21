#!/bin/bash
# STAGE1_CHECKPOINT=<source.pth> [num_agents=2] [num_envs=4] [num_objects=3]
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -gt 3 ]; then
  echo "usage: STAGE1_CHECKPOINT=<source.pth> $0 [agents=2] [envs=4] [objects=3]" >&2
  exit 1
fi
CHECKPOINT=${RESUME_CHECKPOINT:-${STAGE1_CHECKPOINT:-}}
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo 'Set STAGE1_CHECKPOINT (new transfer) or RESUME_CHECKPOINT (bridge Stage 2).' >&2
  exit 1
fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_sm120}
export OUTPUT_PATH=${OUTPUT_PATH:-output/approach_stage2_fan_bridge_watch}
export SENSOR_VIS=${SENSOR_VIS:-1}
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/approach_stage2_fan_bridge_watch.sh" \
  "${1:-2}" "${2:-4}" "${3:-3}"
