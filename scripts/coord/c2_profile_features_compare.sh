#!/bin/bash
# C2 K=1: full speed-profile targets with compact state-derived arrival features.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_profile_features_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

MS_SCEN=cross COORD_MODEL=c2 COORD_C2_DIRECT_SPEED_PROFILE=1 \
    COORD_C2_ARRIVAL_FEATURES=1 COORD_C2_STATE_ORDERED_TARGET_GAP=1 \
    COORD_C2_ARRIVAL_TIME_MARGIN=1.0 COORD_C2_ARRIVAL_GAP_COEF=10.0 \
    COORD_ITERS=500 COORD_SAVE_EVERY=10 COORD_ENVS=64 \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
