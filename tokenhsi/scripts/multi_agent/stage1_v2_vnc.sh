#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "Provide an existing checkpoint.pth; paths are relative to the repository root." >&2
  exit 1
fi
export PORT=${PORT:-6081}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_sm120}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage1_v2/eval}
export TASK_GRAPH=${TASK_GRAPH:-random_scenario}
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/stage1_v2_test.sh" \
  "$1" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
