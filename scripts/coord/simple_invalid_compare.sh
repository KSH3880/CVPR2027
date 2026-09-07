#!/bin/bash
# Track B, one coarse change from B0: close the invalid-plan fallback loophole.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-b1_mlp_invalid1_col50_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

MS_SCEN=cross COORD_MODEL=simple COORD_AUX_COEF=0 \
    COORD_COLLISION_COEF=50 COORD_INVALID_COEF=1 COORD_UNSAFE_COEF=0 \
    COORD_ITERS=500 COORD_SAVE_EVERY=10 COORD_ENVS=64 \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
