#!/bin/bash
# Local 24-GB GPU launcher for stack-planner PPO.
#
# Usage:
#   bash TokenHSI-coord/stack_planner/train_local_gpu1.sh <tag> [executor.pth]
#
# Example smoke:
#   STACK_PLANNER_ENVS=8 STACK_PLANNER_ITERS=1 \
#     STACK_PLANNER_HORIZON=8 STACK_PLANNER_LOW_STEPS=6 \
#     bash TokenHSI-coord/stack_planner/train_local_gpu1.sh local_smoke
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXEC_REPO="$ROOT/TokenHSI-masteer"
TRAIN="$ROOT/TokenHSI-coord/stack_planner/train.sh"
TAG=${1:?usage: train_local_gpu1.sh <tag> [sequential-stack-policy.pth]}
EXEC_CKPT=${2:-"$EXEC_REPO/output/sequential_stack/anti_feat_top_s2/Humanoid_00012000.pth"}

# This wrapper deliberately owns the device selection.  train.sh maps the
# physical device below to logical cuda:0 inside Isaac Gym/PyTorch.
export MA_GPU=1
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118}
export STACK_PLANNER_ENVS=${STACK_PLANNER_ENVS:-1024}
export MS_CKPT=${MS_CKPT:-"$EXEC_REPO/output/ckpt_stage1.pth"}

if [ ! -f "$EXEC_CKPT" ]; then
    echo "local frozen executor checkpoint 없음: $EXEC_CKPT" >&2
    exit 1
fi
if [ ! -f "$MS_CKPT" ]; then
    echo "local stage1 checkpoint 없음: $MS_CKPT" >&2
    exit 1
fi

echo "local stack planner: physical_gpu=1 envs=$STACK_PLANNER_ENVS conda=$TOKENHSI_CONDA_ENV"
exec bash "$TRAIN" "$TAG" "$EXEC_CKPT"
