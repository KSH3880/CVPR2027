#!/bin/bash
# consistency_s7 iter 190: disable exactly one auxiliary, preserving the rest.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOSS=${1:?usage: ablate_loss_r190.sh <curvature|collision|smoothness|speed|consistency|excess|direction> [tag] [iters] [gpu] [--dry-run]}
case "$LOSS" in
    curvature) LOSS_ENV=CARRY_PLANNER_ANALYTIC_CURVATURE_COEF ;;
    collision) LOSS_ENV=CARRY_PLANNER_ANALYTIC_COLLISION_COEF ;;
    smoothness) LOSS_ENV=CARRY_PLANNER_SMOOTHNESS_COEF ;;
    speed) LOSS_ENV=CARRY_PLANNER_SPEED_SMOOTHNESS_COEF ;;
    consistency) LOSS_ENV=CARRY_PLANNER_REPLAN_CONSISTENCY_COEF ;;
    excess) LOSS_ENV=CARRY_PLANNER_EXCESS_LENGTH_COEF ;;
    direction) LOSS_ENV=CARRY_PLANNER_DIRECTION_COEF ;;
    *) echo "unknown auxiliary: $LOSS" >&2; exit 2 ;;
esac
TAG=${2:-carry_implicit_consistency_s7_r190_no_$LOSS}
ITERS=${3:-110}
GPU=${4:-6}
MODE=${5:-run}
case "$TAG" in ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2;; esac
case "$ITERS:$GPU" in *[!0-9:]*) echo 'iterations/GPU must be integers' >&2; exit 2;; esac
[ "$ITERS" -gt 0 ] || exit 2
case "$MODE" in run|--dry-run) ;; *) echo 'fifth argument: --dry-run or run' >&2; exit 2;; esac
SOURCE_RUN="$ROOT/runs/carry_planner/carry_implicit_consistency_s7"
INIT="$SOURCE_RUN/planner_000190.pth"
for FILE in "$SOURCE_RUN/run.env" "$INIT"; do
    [ -f "$FILE" ] || { echo "missing: $FILE" >&2; exit 1; }
done
[ ! -e "$ROOT/runs/carry_planner/$TAG" ] || { echo "existing run: $TAG" >&2; exit 3; }
set -a
. "$SOURCE_RUN/run.env"
set +a
export MS_CKPT="$stage1" MA_GPU="$GPU"
export CARRY_PLANNER_INIT="$INIT" CARRY_PLANNER_ITERS="$ITERS"
export CARRY_PLANNER_SEED=0 CARRY_PLANNER_ENVS=2048
ORIGINAL_COEF=${!LOSS_ENV}
export "$LOSS_ENV=0"
export CARRY_PLANNER_OUTPUT="$ROOT/runs/carry_planner/$TAG"
export PYTHONDONTWRITEBYTECODE=1
# Use the existing per-GPU MPS server; never start or stop a daemon here.
. "$ROOT/mps/shell.sh"
mps_use "$GPU"
echo "tag=$TAG init=$INIT iterations=191..$((190 + ITERS)) gpu=$GPU"
echo "reward collision=$CARRY_PLANNER_COLLISION_COEF invalid=$CARRY_PLANNER_INVALID_PLAN_COEF; PPO lr=$CARRY_PLANNER_LR clip=$CARRY_PLANNER_CLIP"
echo "ablation: $LOSS_ENV=$ORIGINAL_COEF -> 0; other auxiliaries and PPO/reward preserved; KL measurement enabled"
if [ "$MODE" = --dry-run ]; then
    env | LC_ALL=C sort | rg '^(CARRY_PLANNER_|CUDA_MPS_PIPE_DIRECTORY=|MS_CKPT=)'
    exit 0
fi
exec bash "$ROOT/TokenHSI-coord/carry_planner/train.sh" "$TAG" "$executor"
