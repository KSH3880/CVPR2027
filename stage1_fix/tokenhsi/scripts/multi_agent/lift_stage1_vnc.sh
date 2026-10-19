#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export PORT=${PORT:-6081}
export HEADLESS=0
lift_gpu_index=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
[ "$lift_gpu_index" = 6 ] || { echo 'Lift uses physical GPU 6 only' >&2; exit 1; }
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/lift_stage1_test.sh" "$@"
