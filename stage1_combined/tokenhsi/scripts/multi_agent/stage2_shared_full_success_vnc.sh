#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
  echo 'Provide an existing Stage2 checkpoint.pth' >&2
  exit 1
fi
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
export PORT=${PORT:-6081}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage2_shared_full_success/eval}
export HEADLESS=0
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/stage2_shared_full_success_test.sh" "$1" "${2:-4}" "${3:-1}" "${4:-12}" "${5:-10}"
