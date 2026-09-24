#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "Provide an existing checkpoint.pth; paths are relative to the repository root." >&2
  exit 1
fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output/approach_independent_climb_placement_region_vnc}
export TASK_GRAPH=${TASK_GRAPH:-climb_ontop}
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/approach_independent_climb_placement_region_test.sh" \
  "$1" "${2:-2}" "${3:-1}" "${4:-3}" "${5:-10}"
