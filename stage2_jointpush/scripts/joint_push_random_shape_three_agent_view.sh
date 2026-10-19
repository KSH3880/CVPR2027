#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export JOINT_PUSH_THREE_AGENT_PREVIEW=1 NUM_AGENTS=3
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/random_shape_three_agent_view}
exec bash "$ROOT/scripts/joint_push_random_shape_view.sh" "$@"
