#!/bin/bash
# C2 K=1 toy baseline: one flat joint MLP predicts residual path and speed profile.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_mlp_profile_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

# The long-running feature queue may invoke this script again after an early
# parallel comparison has already completed.  A full 500-step checkpoint is
# authoritative and makes the second invocation a no-op.
if [ -f "$ROOT/runs/coord/$TAG/coord_c2_000500.pth" ]; then
    echo "C2_MLP_REUSE_COMPLETE tag=$TAG"
    exit 0
fi

MS_SCEN=cross COORD_MODEL=c2 COORD_C2_MLP_BACKBONE=1 \
    COORD_C2_DIRECT_SPEED_PROFILE=1 \
    COORD_C2_STATE_ORDERED_TARGET_GAP=1 \
    COORD_C2_PATH_RESIDUAL_COEF=1.0 \
    COORD_C2_ARRIVAL_TIME_MARGIN=1.0 COORD_C2_ARRIVAL_GAP_COEF=10.0 \
    COORD_ITERS=500 COORD_SAVE_EVERY=10 COORD_ENVS=64 \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
