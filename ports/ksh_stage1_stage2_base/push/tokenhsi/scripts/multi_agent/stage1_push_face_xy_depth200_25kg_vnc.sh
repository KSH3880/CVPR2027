#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/stage1_push_face_xy_depth200_25kg_test.sh" "$@"
