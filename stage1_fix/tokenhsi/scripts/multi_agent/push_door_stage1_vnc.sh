#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ ! -f "$1" ]; then echo 'Provide an existing push/door checkpoint.pth.' >&2; exit 1; fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-output/push_door_stage1_viewer}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/push_door_stage1_test.sh" "$1" "${2:-1}" "${3:-3}"
