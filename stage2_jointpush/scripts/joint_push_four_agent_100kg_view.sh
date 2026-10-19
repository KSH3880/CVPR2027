#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export JOINT_PUSH_THREE_AGENT_PREVIEW=1 NUM_AGENTS=4
export JOINT_PUSH_PREVIEW_BOX_SIZE=3,1.5,1.5 JOINT_PUSH_PREVIEW_BOX_MASS=100
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/four_agent_100kg_view}
exec bash "$ROOT/scripts/joint_push_random_shape_view.sh" "$@"
