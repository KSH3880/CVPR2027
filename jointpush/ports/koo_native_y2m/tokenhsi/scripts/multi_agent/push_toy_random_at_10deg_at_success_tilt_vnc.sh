#!/bin/bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
  echo "Provide an existing checkpoint.pth; paths are relative to the repository root." >&2
  exit 1
fi
if [ -z "${TOKENHSI_CONDA_ENV:-}" ]; then
  case "${CONDA_DEFAULT_ENV:-}" in
    ''|base) export TOKENHSI_CONDA_ENV=tokenhsi_koo ;;
    *) export TOKENHSI_CONDA_ENV=$CONDA_DEFAULT_ENV ;;
  esac
fi
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output/push_toy_random_at_10deg_at_success_tilt}
export HEADLESS=0
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/push_toy_random_at_10deg_at_success_tilt_test.sh" \
  "$1" "${2:-2}" "${3:-1}" "${4:-2}" "${5:-10}"
