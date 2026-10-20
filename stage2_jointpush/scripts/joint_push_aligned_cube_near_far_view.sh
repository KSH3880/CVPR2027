#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-6} PORT=${PORT:-6086}
export NUM_AGENTS=2 NUM_ENVS=1
unset JOINT_PUSH_VIEW_GRID_INDEX JOINT_PUSH_THREE_AGENT_PREVIEW JOINT_PUSH_VIEW_MASS
export JOINT_PUSH_TWO_AGENT_PAYLOAD_PREVIEW=1
export JOINT_PUSH_PREVIEW_BOX_SIZE=2.6,2.6,2.6 JOINT_PUSH_PREVIEW_BOX_MASS=60
export JOINT_PUSH_VIEW_SPAWN_DISTANCES=${JOINT_PUSH_VIEW_SPAWN_DISTANCES:-0.8,3.0}
export JOINT_PUSH_REQUIRE_ALIGNED=1
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/aligned_cube_near_far_view}
CKPT=${1:-$ROOT/../runs/jointpush/random_shape_resume_gpu6_20261015_223106/training/JointPushRandomShapeMassSoftGateCa4_15-22-31-17/nn/JointPushRandomShapeMassSoftGateCa4_00028000.pth}
exec bash "$ROOT/scripts/joint_push_random_shape_view.sh" "$CKPT"
