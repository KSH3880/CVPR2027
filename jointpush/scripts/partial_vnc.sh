#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export TOKENHSI_GUI_GPU_MODE=native HEADLESS=0
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc} PORT=${PORT:-6086} RESOLUTION=${RESOLUTION:-1280x720x24}
exec bash "$ROOT/jointpush/scripts/run_gui.sh" bash "$ROOT/jointpush/scripts/partial_test.sh" "$@"
