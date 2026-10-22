#!/bin/bash
set -eu
source "$(dirname "$0")/koo_1021_common.sh"
koo_checkpoint "${1:-}" train
: "${TOKENHSI_GPU:?Set TOKENHSI_GPU explicitly}"
export RESUME_CHECKPOINT="$CHECKPOINT"
export OUTPUT_PATH=${OUTPUT_PATH:-$KOO_PORT/output/stage1_base_push_training_$(date +%Y%m%d_%H%M%S)}
exec bash "$KOO_PROJECT/jointpush/scripts/koo_1021_runtime.sh" train 2 2048 4
