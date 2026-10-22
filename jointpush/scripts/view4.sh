#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU to the intended physical GPU}"
preflight
export TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1
export HEADLESS=0 RESOLUTION=${RESOLUTION:-1280x720x24} PORT=${PORT:-6086}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/y2m/view4_$(date +%Y%m%d_%H%M%S)}
exec bash "$JOINTPUSH_ROOT/scripts/run_gui.sh" bash "$JOINTPUSH_ROOT/scripts/view4_run.sh" "$CHECKPOINT"
