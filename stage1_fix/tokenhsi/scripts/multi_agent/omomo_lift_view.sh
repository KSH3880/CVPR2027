#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
. "$SCRIPT_DIR/gui_gpu_env.sh"
[ "$TOKENHSI_GUI_GPU_INDEX" = 6 ] || { echo "OMOMO viewer requires physical GPU 6" >&2; exit 1; }
. "$SCRIPT_DIR/runtime_env.sh"
exec python tokenhsi/scripts/multi_agent/omomo_lift_preview.py "$@"
