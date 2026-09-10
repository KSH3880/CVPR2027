#!/bin/bash
# C10: C9 joint-both baseline plus learned yielder concentration.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c10_dualdelay_s0}
DUAL=${2:-1.0}
EXECUTOR=${3:-ms18_maskteam_origscale_c06_s0}

COORD_C2_DUAL_DELAY_COEF="$DUAL" \
    COORD_ITERS=${COORD_ITERS:-300} COORD_SAVE_EVERY=${COORD_SAVE_EVERY:-100} \
    bash "$ROOT/scripts/coord/c9_jointboth_c5base.sh" "$TAG" "$EXECUTOR"
