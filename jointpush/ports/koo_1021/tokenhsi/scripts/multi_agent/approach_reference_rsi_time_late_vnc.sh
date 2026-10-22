#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_REPO=stage1_fix TOKENHSI_GPU=${TOKENHSI_GPU:-6} HEADLESS=0
exec bash "$SCRIPT_DIR/../../../../scripts/tokenhsi_gui.sh" --gpu "$TOKENHSI_GPU" --port "${PORT:-6110}" \
  bash "$SCRIPT_DIR/approach_reference_rsi_time_late_test.sh" "${1:?Provide this experiment checkpoint.pth}"
