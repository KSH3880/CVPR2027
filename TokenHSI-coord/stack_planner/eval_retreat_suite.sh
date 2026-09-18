#!/bin/bash
# Evaluate one planner on randomized free scenes and three controlled layouts.
# Usage: bash .../eval_retreat_suite.sh <training-tag> <planner.pth> <eval-prefix>
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TRAIN_TAG=${1:?training tag required}
PLANNER=${2:?planner checkpoint required}
PREFIX=${3:?unique evaluation prefix required}

for SCEN in free cross parallel solo; do
    STACK_PLANNER_EVAL_SCEN="$SCEN" \
    STACK_PLANNER_EVAL_BOX_GRID=${STACK_PLANNER_EVAL_BOX_GRID:-1} \
    bash "$ROOT/TokenHSI-coord/stack_planner/eval_retreat.sh" \
        "$TRAIN_TAG" "$PLANNER" "${PREFIX}_${SCEN}"
done
