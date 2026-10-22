#!/bin/bash
set -eu
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=''
exec "$JOINTPUSH_PYTHON" "$JOINTPUSH_ROOT/scripts/prepare.py" --report "$PROJECT_ROOT/runs/jointpush/y2m/preparation.json" "$@"
