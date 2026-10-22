#!/bin/bash
set -eu
scripts=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$scripts/../../.." && pwd)
cd "$root"
[ $# -ge 1 ] && [ -f "$1" ] || { echo 'Provide a Stage1BasePush checkpoint' >&2; exit 1; }
checkpoint=$(realpath -e "$1")
export TOKENHSI_GPU=${TOKENHSI_GPU:-1}
export TOKENHSI_GUI_GPU_MODE=${TOKENHSI_GUI_GPU_MODE:-native}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
export TOKENHSI_VIEWER_LOW_VRAM=${TOKENHSI_VIEWER_LOW_VRAM:-1}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export RESOLUTION=${RESOLUTION:-1280x720x24}
export PORT=${PORT:-6107}
export HEADLESS=0
exec bash "$scripts/run-gui.sh" bash "$scripts/stage1_base_push_test.sh" \
  "$checkpoint" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
