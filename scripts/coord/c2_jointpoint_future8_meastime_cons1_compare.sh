#!/bin/bash
# Measured-timing C2 with random fixed priority and trajectory consistency.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_jointpoint_future8_meastime_rprio_cons1_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

if [ -f "$ROOT/runs/coord/$TAG/coord_c2_000500.pth" ]; then
    echo "C2_FUTURE8_MEASTIME_CONS1_REUSE_COMPLETE tag=$TAG"
    exit 0
fi

LOCK_DIR="$ROOT/runs/queue/locks/${TAG}.launch_lock"
mkdir -p "$(dirname -- "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "C2_FUTURE8_MEASTIME_CONS1_REUSE_RUNNING tag=$TAG"
    while pgrep -f "[c]oordinator.train_closed_loop.*runs/coord/$TAG" >/dev/null; do
        sleep 20
    done
    test -f "$ROOT/runs/coord/$TAG/coord_c2_000500.pth"
    exit $?
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

MS_SCEN=cross COORD_MODEL=c2 COORD_C2_MLP_BACKBONE=1 \
    COORD_C2_MLP_CONFLICT_FEATURES=0 \
    COORD_C2_DIRECT_WAYPOINTS=1 COORD_C2_DIRECT_SPEED_PROFILE=1 \
    COORD_C2_JOINT_POINT_SPEED=1 COORD_C2_WAYPOINT_SMOOTHING_PASSES=8 \
    COORD_C2_WAYPOINT_DISTANCE_SCALING=1 \
    COORD_C2_STATE_ORDERED_TARGET_GAP=0 COORD_C2_FIXED_YIELD_SPEED=0 \
    COORD_C2_PATH_COLLISION_ONLY=0 COORD_C2_COLLISION_FOCUS_STEPS=8 \
    COORD_C2_COLLISION_TIME_UNCERTAINTY=0 \
    COORD_C2_MEASURED_EXECUTOR_TIMING=1 \
    COORD_C2_RANDOM_PRIORITY=1 \
    COORD_C2_CONSISTENCY_COEF=1 \
    COORD_C2_FUTURE_COLLISION_COEF=50 \
    COORD_C2_SPEED_EFFICIENCY_COEF=0.1 \
    COORD_C2_PATH_RESIDUAL_COEF=1.0 COORD_C2_PATH_SMOOTH_COEF=10.0 \
    COORD_ITERS=${COORD_ITERS:-500} COORD_SAVE_EVERY=${COORD_SAVE_EVERY:-10} \
    COORD_ENVS=${COORD_ENVS:-64} \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

if [ "${COORD_SKIP_FINAL_EVAL:-0}" != 1 ]; then
    bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
    bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
fi
