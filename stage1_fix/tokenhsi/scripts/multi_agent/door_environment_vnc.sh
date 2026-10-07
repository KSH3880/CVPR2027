#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
export NUM_ENVS=${NUM_ENVS:-1}
export REPEATS=${REPEATS:-3}
export OUTPUT_PATH=${OUTPUT_PATH:-output/door_environment_viewer}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/door_environment_test.sh" "$@"
