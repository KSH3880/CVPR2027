#!/bin/bash
# C8: C7 joint-both risk model with the known-stable path residual weight.
# Keep this as a single-variable diagnostic before adding any new loss term.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c8_joint_risk_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

COORD_C2_PATH_RESIDUAL_COEF=1 \
    COORD_ITERS=${COORD_ITERS:-300} \
    COORD_SAVE_EVERY=${COORD_SAVE_EVERY:-100} \
    bash "$ROOT/scripts/coord/c7_jointboth.sh" "$TAG" "$EXECUTOR"
