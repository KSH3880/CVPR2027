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
export OUTPUT_PATH=${OUTPUT_PATH:-output/open_toy_hand_contact}
export HEADLESS=0
exec bash "$SCRIPT_DIR/run-gui.sh" \
  bash "$SCRIPT_DIR/open_toy_test.sh" \
  "$1" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
