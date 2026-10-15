#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export JOINT_PUSH_VIEW_MASS=40
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/softgate_mass40_view}
export PORT=${PORT:-6086}
exec bash "$ROOT/scripts/joint_push_softgate_view.sh" "$@"
