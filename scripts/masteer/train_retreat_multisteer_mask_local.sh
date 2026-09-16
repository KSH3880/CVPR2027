#!/bin/bash
# Retrain ms58's shared-goal sequence with diverse CLEAR steering and a smooth
# carry-observation fade for the released base carrier.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms61_ms18e9000_retreatrand_zerofade_move80_pathpen0_3000_s0}

# Reuse the ms58 recipe, overriding only this experiment's isolated axis.
export PILOT_GPU=${PILOT_GPU:-1}
if [[ "$PILOT_GPU" != 1 ]]; then
    echo "이 로컬 실행 래퍼는 GPU 1만 허용한다: $PILOT_GPU" >&2
    exit 2
fi
export PILOT_ENVS=${PILOT_ENVS:-1024}
export PILOT_ITERS=${PILOT_ITERS:-3000}
export PILOT_RETREAT_SIDE_DEG=90
export PILOT_RETREAT_SIDE_BINS=1
export PILOT_RETREAT_RANDOM=1
export PILOT_RETREAT_DIST=${PILOT_RETREAT_DIST:-1.5}
export PILOT_RETREAT_DIST_MIN=${PILOT_RETREAT_DIST_MIN:-0.8}
export PILOT_CLEAR_ARC_DIST=0.60
export PILOT_VIRTUAL_RETREAT_BOX=0
export PILOT_VIRTUAL_RETREAT_REAR_BOX=0
export PILOT_ZERO_CARRY_OBS=0
export PILOT_DYNAMIC_CARRY_MASK=0
export PILOT_CARRY_OBS_ZERO_FADE=1
# Restore ms52's zero path-penalty contract while strongly rewarding actual
# on-command retreat. This removes the incentive to avoid entering CLEAR just
# to escape a large negative reward.
export PILOT_CLEAR_MOVE_W=${PILOT_CLEAR_MOVE_W:-80.0}
export PILOT_CLEAR_STALL_PEN_W=0.0
export PILOT_CLEAR_PATH_PEN_W=0.0

exec bash "$ROOT/scripts/masteer/train_ms52_shared_goal_safe_local.sh" "$TAG"
