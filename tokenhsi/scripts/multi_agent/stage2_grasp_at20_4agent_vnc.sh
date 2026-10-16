#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "Provide an existing Stage-2 checkpoint.pth; paths are relative to the repository root." >&2
  exit 1
fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output/stage2_grasp_at20_4agent}
export TASK_GRAPH=${TASK_GRAPH:-random_scenario}
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/stage2_grasp_at20_4agent_test.sh" \
  "$1" "${2:-4}" "${3:-1}" "${4:-8}" "${5:-10}"
