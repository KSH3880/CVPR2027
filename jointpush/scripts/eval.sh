#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
resolve_checkpoint "${1:-}"
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU to the intended physical GPU}"
preflight
export HEADLESS=1
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/y2m/evaluation_$(date +%Y%m%d_%H%M%S)}
exec bash "$PORT_ROOT/tokenhsi/scripts/multi_agent/push_native_phase_y2m_test.sh" "$CHECKPOINT" 2 "${NUM_ENVS:-4}" 2 "${REPEATS:-3}"
