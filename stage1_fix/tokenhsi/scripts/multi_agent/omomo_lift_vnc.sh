#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export PORT=${PORT:-6080}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/omomo_lift_view.sh" "$@"
