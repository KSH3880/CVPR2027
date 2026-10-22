#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export JOINTPUSH_VIEW_BIAS=1 JOINTPUSH_BIAS_CORRECTION=1
export JOINTPUSH_BIAS_SCOPE=all_owner JOINTPUSH_BIAS_TARGET=all_foreign
export JOINTPUSH_BIAS_LAYERS=all JOINTPUSH_BIAS_LAMBDA=4 JOINTPUSH_BIAS_HARD_MASK=0
export JOINTPUSH_BOXES=4
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/y2m/view4_bias_all4_$(date +%Y%m%d_%H%M%S)}
exec bash "$ROOT/scripts/view4.sh" "$@"
