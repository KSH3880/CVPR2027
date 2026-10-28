#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export PORT=${PORT:-6109}
export HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage1_jointpush_density3_cone90_finetune_atprogress_gate/eval_vnc}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/stage1_jointpush_density3_cone90_finetune_atprogress_gate_test.sh" \
  "${1:?Pass the checkpoint trained with the AT progress gate config}" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
