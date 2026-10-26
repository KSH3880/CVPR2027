#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export PORT=${PORT:-6089}
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130/eval_vnc}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130_test.sh" \
  "${1:?Provide a PUSH cuboid checkpoint}" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}" "${6:-}"
