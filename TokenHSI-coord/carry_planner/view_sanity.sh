#!/bin/bash
# View the deterministic untrained V16 spline profile with the frozen ms18 executor.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PLANNER=${CARRY_PLANNER_SANITY_CKPT:-"$ROOT/runs/carry_planner/sanity_sparse_v16/planner_detour_slow.pth"}
EXECUTOR=${1:-"$ROOT/TokenHSI-masteer/output/ms18_maskteam_origscale_c06_s0_00009000.pth"}

[ -f "$PLANNER" ] || {
    echo "sanity planner checkpoint 없음: $PLANNER" >&2
    echo "make_sanity_checkpoint.py로 먼저 생성한다" >&2
    exit 2
}
[ -f "$EXECUTOR" ] || { echo "frozen executor checkpoint 없음: $EXECUTOR" >&2; exit 2; }

export CARRY_PLANNER_CONVERGE_PROB=${CARRY_PLANNER_CONVERGE_PROB:-0}
export MS_SCEN=${MS_SCEN:-cross}
export MS_VIEW_TIMED_CROSS=${MS_VIEW_TIMED_CROSS:-1}
export CARRY_PLANNER_DEBUG=${CARRY_PLANNER_DEBUG:-1}
exec bash "$ROOT/TokenHSI-coord/carry_planner/view.sh" "$PLANNER" "$EXECUTOR"
