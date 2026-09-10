#!/bin/bash
# Explicit behavior shaping: assigned yielder, box-attached interval, restore after crossing.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_slowwin_explicit_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

COORD_C2_EXPLICIT_GAP_COEF=10 \
COORD_C2_EXPLICIT_ANCHOR_COEF=3 \
COORD_C2_EXPLICIT_POST_COEF=3 \
COORD_C2_INITIAL_PLAN_WEIGHT=3 \
    bash "$ROOT/scripts/coord/c2_queue_slowwin.sh" \
        "$TAG" 0 "$EXECUTOR"
