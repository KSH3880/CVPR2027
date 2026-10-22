#!/bin/bash
set -eu
source "$(dirname "$0")/koo_1021_common.sh"
koo_checkpoint "${1:-}" test
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
export TOKENHSI_GUI_GPU_MODE=native HEADLESS=0 TOKENHSI_VIEWER_LOW_VRAM=1
export PORT=${PORT:-6086} RESOLUTION=${RESOLUTION:-1280x720x24}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-$KOO_PORT/output/stage1_base_push_view_$(date +%Y%m%d_%H%M%S)}
exec bash "$KOO_PROJECT/jointpush/scripts/run_gui.sh" bash "$KOO_PROJECT/jointpush/scripts/koo_1021_runtime.sh" test "$CHECKPOINT" 2 "${NUM_ENVS:-1}" 4 "${REPEATS:-10000}"
