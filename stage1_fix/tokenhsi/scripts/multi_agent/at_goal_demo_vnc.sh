#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-output/at_goal_demo}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/at_goal_demo_test.sh" "$@"
