#!/bin/bash
set -eu
source "$(dirname "$0")/koo_1021_common.sh"
koo_checkpoint "${1:-}" test
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
export HEADLESS=${HEADLESS:-1}
if [ "$HEADLESS" = 0 ]; then export TOKENHSI_GUI_GPU_MODE=native; fi
export OUTPUT_PATH=${OUTPUT_PATH:-$KOO_PORT/output/stage1_base_push_eval_$(date +%Y%m%d_%H%M%S)}
exec bash "$KOO_PROJECT/jointpush/scripts/koo_1021_runtime.sh" test "$CHECKPOINT" 2 "${NUM_ENVS:-1}" 4 "${REPEATS:-10}"
