#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
export VIRTUAL_CFG_ENV=tokenhsi/data/cfg/multi_agent/joint_push_virtual_curve_stage1.yaml
export OUTPUT_PATH=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/virtual_curve_view}
export PORT=${PORT:-6084}
exec bash "$ROOT/scripts/joint_push_virtual_view.sh" "$@"
