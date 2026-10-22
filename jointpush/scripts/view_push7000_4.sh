#!/bin/bash
set -eu
if [ "${1:-}" = --help ]; then
 echo 'TOKENHSI_GPU=5 PORT=6086 bash jointpush/scripts/view_push7000_4.sh [checkpoint]'
 echo 'Original 7000 PUSH policy; 4 agents, 8 boxes; no added bias or masking.'
 exit 0
fi
source "$(dirname "$0")/common.sh"
: "${TOKENHSI_GPU:?Set the intended physical GPU explicitly}"
export FIVE_TASK_VIEW_WORK="$PROJECT_ROOT/runs/jointpush/five_task_push7000_scaling_20261022/workspace"
DEFAULT_CHECKPOINT=$("$JOINTPUSH_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$FIVE_TASK_VIEW_WORK/preparation.json")
resolve_checkpoint "${1:-$DEFAULT_CHECKPOINT}"
export FIVE_TASK_VIEW_CHECKPOINT="$CHECKPOINT" PROBE_AGENTS=4
export TOKENHSI_GUI_GPU_MODE=native TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0
export PORT=${PORT:-6086} RESOLUTION=${RESOLUTION:-1280x720x24}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export WEBSOCKIFY=${WEBSOCKIFY:-$HOME/anaconda3/envs/$TOKENHSI_CONDA_ENV/bin/websockify}
export OUTPUT_PATH=${OUTPUT_PATH:-$FIVE_TASK_VIEW_WORK/output_etc/check/view4_$(date +%Y%m%d_%H%M%S)}
exec bash "$JOINTPUSH_ROOT/scripts/run_gui.sh" bash "$JOINTPUSH_ROOT/scripts/view_push7000_4_run.sh"
