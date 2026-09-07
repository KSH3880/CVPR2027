#!/bin/bash
# Same direct crossing-time-gap objective, with one coarse 10x strength change.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_timegap20_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

MS_SCEN=cross COORD_MODEL=c2 COORD_C2_IMMEDIATE_SLOWDOWN=1 \
    COORD_C2_ARRIVAL_GAP=1 COORD_C2_ARRIVAL_TIME_MARGIN=1.0 \
    COORD_C2_ARRIVAL_GAP_COEF=20.0 \
    COORD_ITERS=500 COORD_SAVE_EVERY=10 COORD_ENVS=64 \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
