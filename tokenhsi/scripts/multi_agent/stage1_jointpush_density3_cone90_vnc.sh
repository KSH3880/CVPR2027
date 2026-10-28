#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export PORT=${PORT:-6103}
export HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage1_jointpush_density3_cone90/eval_vnc}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/stage1_jointpush_density3_cone90_test.sh" \
  "${1:-pth/Stage1PushFinetune_00014000.pth}" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
