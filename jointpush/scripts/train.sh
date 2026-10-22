#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU to an available physical GPU}"
preflight
export RESUME_CHECKPOINT="$CHECKPOINT"
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/y2m/training_$(date +%Y%m%d_%H%M%S)}
exec bash "$PORT_ROOT/tokenhsi/scripts/multi_agent/push_native_phase_y2m_train.sh" 2 2048 2
