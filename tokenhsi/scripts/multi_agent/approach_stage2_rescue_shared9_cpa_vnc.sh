#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "Provide an existing checkpoint.pth (STAGE1_ONLY=1 for Stage-1); paths are relative to the repository root." >&2
  exit 1
fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
DEFAULT_OUTPUT=output/approach_stage2_rescue_shared9_cpa
if [ "${STAGE1_ONLY:-0}" = 1 ]; then DEFAULT_OUTPUT=$DEFAULT_OUTPUT/stage1_only; fi
export OUTPUT_PATH=${OUTPUT_PATH:-$DEFAULT_OUTPUT}
export TASK_GRAPH=${TASK_GRAPH:-random_scenario}
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/approach_stage2_rescue_shared9_cpa_test.sh" \
  "$1" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
