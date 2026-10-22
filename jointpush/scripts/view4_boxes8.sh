#!/bin/bash
set -eu
export JOINTPUSH_BOXES=8
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/y2m/view4_boxes8_$(date +%Y%m%d_%H%M%S)}
exec bash "$ROOT/scripts/view4.sh" "$@"
