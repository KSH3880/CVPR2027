#!/bin/bash
# C2 joint space-speed trajectory: one MLP head emits (x,y,v) at all path points.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_jointpoint_mlp_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

if [ -f "$ROOT/runs/coord/$TAG/coord_c2_000500.pth" ]; then
    echo "C2_JOINTPOINT_REUSE_COMPLETE tag=$TAG"
    exit 0
fi

# A pre-existing delayed queue may reach this same tag after an early parallel
# launch. Atomically own the tag so two PPO writers can never share one run.
LOCK_DIR="$ROOT/runs/queue/locks/${TAG}.launch_lock"
mkdir -p "$(dirname -- "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "C2_JOINTPOINT_REUSE_RUNNING tag=$TAG"
    while pgrep -f "[c]oordinator.train_closed_loop.*runs/coord/$TAG" >/dev/null; do
        sleep 20
    done
    test -f "$ROOT/runs/coord/$TAG/coord_c2_000500.pth"
    exit $?
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

MS_SCEN=cross COORD_MODEL=c2 COORD_C2_MLP_BACKBONE=1 \
    COORD_C2_DIRECT_WAYPOINTS=1 COORD_C2_DIRECT_SPEED_PROFILE=1 \
    COORD_C2_JOINT_POINT_SPEED=1 COORD_C2_STATE_ORDERED_TARGET_GAP=1 \
    COORD_C2_PATH_RESIDUAL_COEF=1.0 COORD_C2_PATH_SMOOTH_COEF=10.0 \
    COORD_C2_ARRIVAL_TIME_MARGIN=1.0 COORD_C2_ARRIVAL_GAP_COEF=10.0 \
    COORD_ITERS=500 COORD_SAVE_EVERY=10 COORD_ENVS=64 \
    bash "$ROOT/scripts/coord/train_local.sh" "$TAG" "$EXECUTOR"

bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" cross 128 0
bash "$ROOT/scripts/coord/eval_one.sh" "$TAG" free 128 0
