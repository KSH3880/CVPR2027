#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_GUI_GPU_MODE=native HEADLESS=0
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118}
export PORT=${PORT:-6086} VNC_DIR=${VNC_DIR:-$HOME/opt/vnc} RESOLUTION=${RESOLUTION:-1280x720x24}
if [ -n "${1:-}" ]; then export CHECKPOINT_OVERRIDE=$1; fi
exec bash "$ROOT/jointpush/scripts/run_gui.sh" bash "$ROOT/ksh_jointpush/scripts/ksh_jointpush_launch.sh" view
