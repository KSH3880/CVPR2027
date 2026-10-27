#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_jhh}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export PORT=${PORT:-6081}
export HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage2_jointpush_rsi/eval_vnc}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/stage2_jointpush_rsi_test.sh" \
  "${1:-pth/Stage1PushFinetune_00010500.pth}" "${2:-4}" "${3:-1}" "${4:-8}" "${5:-10}"
