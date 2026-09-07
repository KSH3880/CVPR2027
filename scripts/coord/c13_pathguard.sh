#!/bin/bash
# C13: C12 random-role min-risk objective with stronger minimal-path prior.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c13_path10_s0}
PATH_COEF=${2:-10}
EXECUTOR=${3:-ms18_maskteam_origscale_c06_s0}

COORD_C2_PATH_RESIDUAL_COEF="$PATH_COEF" \
    bash "$ROOT/scripts/coord/c12_minrisk.sh" \
        "$TAG" random "$EXECUTOR"
