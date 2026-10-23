#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "Provide an existing warmstart checkpoint.pth" >&2
  exit 1
fi
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/open_push_zero_shot_random_goal_test.sh" \
  "$1" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-100}"
