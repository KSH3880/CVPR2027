#!/bin/bash
# Compare compact slowdown-window policies. WIDTH_MAX=0 allows the full path.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: c2_slowwin_compare.sh <tag> <width-max-m> [executor]}
WIDTH_MAX=${2:?usage: c2_slowwin_compare.sh <tag> <width-max-m> [executor]}
EXECUTOR=${3:-ms18_maskteam_origscale_c06_s0}
MAX_ITER=${COORD_ITERS:-250}

if ! [[ "$WIDTH_MAX" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "width-max는 0 이상의 숫자여야 한다: $WIDTH_MAX" >&2
    exit 2
fi
printf -v padded '%06d' "$MAX_ITER"
if [ -f "$ROOT/runs/coord/$TAG/coord_c2_${padded}.pth" ]; then
    echo "C2_SLOWWIN_REUSE_COMPLETE tag=$TAG step=$MAX_ITER width_max=$WIDTH_MAX"
    exit 0
fi

LOCK_DIR="$ROOT/runs/queue/locks/${TAG}.launch_lock"
mkdir -p "$(dirname -- "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "C2_SLOWWIN_REUSE_RUNNING tag=$TAG"
    while pgrep -f "[c]oordinator.train_closed_loop.*runs/coord/$TAG" >/dev/null; do
        sleep 20
    done
    test -f "$ROOT/runs/coord/$TAG/coord_c2_${padded}.pth"
    exit $?
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

MS_SCEN=cross COORD_MODEL=c2 COORD_C2_MLP_BACKBONE=1 \
    COORD_C2_MLP_CONFLICT_FEATURES=0 \
    COORD_C2_DIRECT_WAYPOINTS=1 COORD_C2_DIRECT_SPEED_PROFILE=1 \
    COORD_C2_JOINT_POINT_SPEED=0 COORD_C2_SLOWDOWN_WINDOW=1 \
    COORD_C2_SLOWDOWN_WIDTH_MAX="$WIDTH_MAX" \
    COORD_C2_WAYPOINT_SMOOTHING_PASSES=8 \
    COORD_C2_WAYPOINT_DISTANCE_SCALING=1 \
    COORD_C2_STATE_ORDERED_TARGET_GAP=0 COORD_C2_FIXED_YIELD_SPEED=0 \
    COORD_C2_PATH_COLLISION_ONLY=0 COORD_C2_COLLISION_FOCUS_STEPS=8 \
    COORD_C2_COLLISION_TIME_UNCERTAINTY=0 \
    COORD_C2_MEASURED_EXECUTOR_TIMING=1 COORD_C2_RANDOM_PRIORITY=1 \
    COORD_C2_CONSISTENCY_COEF=1 COORD_C2_FUTURE_COLLISION_COEF=50 \
    COORD_C2_SPEED_SMOOTH_COEF=0 COORD_C2_UNNECESSARY_SLOW_COEF=0 \
    COORD_C2_SPEED_EFFICIENCY_COEF=0 COORD_C2_EXTRA_DELAY_COEF=1 \
    COORD_C2_EXPLICIT_GAP_COEF=${COORD_C2_EXPLICIT_GAP_COEF:-0} \
    COORD_C2_EXPLICIT_ANCHOR_COEF=${COORD_C2_EXPLICIT_ANCHOR_COEF:-0} \
    COORD_C2_EXPLICIT_POST_COEF=${COORD_C2_EXPLICIT_POST_COEF:-0} \
    COORD_C2_INITIAL_PLAN_WEIGHT=${COORD_C2_INITIAL_PLAN_WEIGHT:-1} \
    COORD_C2_PATH_RESIDUAL_COEF=1 COORD_C2_PATH_SMOOTH_COEF=0 \
    COORD_ITERS="$MAX_ITER" COORD_SAVE_EVERY=${COORD_SAVE_EVERY:-10} \
    COORD_ENVS=${COORD_ENVS:-64} \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

if [ "${COORD_SKIP_FINAL_EVAL:-0}" != 1 ]; then
    bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
    bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
fi
