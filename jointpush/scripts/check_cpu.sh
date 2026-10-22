#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
if [ -n "${1:-}" ]; then
 resolve_checkpoint "$1"
 preflight
 export TOKENHSI_CHECKPOINT="$CHECKPOINT"
else
 unset TOKENHSI_CHECKPOINT
fi
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/y2m/cpu_check_$(date +%Y%m%d_%H%M%S)}
exec bash "$PORT_ROOT/tokenhsi/scripts/multi_agent/check_push_native_y2m_scene.sh"
