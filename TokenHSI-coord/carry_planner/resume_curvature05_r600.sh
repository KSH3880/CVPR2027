#!/bin/bash
# Resume curvature=0.5 from iteration 600 through 3000.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-carry_implicit_curvature05_lr3e4_clip20_3000_s0_r600_resume}
GPU=${2:-6}
MODE=${3:-run}
case "$TAG" in ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2;; esac
case "$GPU" in ""|*[!0-9]*) echo 'GPU must be a non-negative integer' >&2; exit 2;; esac
case "$MODE" in run|--dry-run) ;; *) echo 'third argument: run or --dry-run' >&2; exit 2;; esac
SOURCE_ENV="$ROOT/runs/carry_planner/carry_implicit_curvature05_lr3e4_clip20_3000_s0/run.env"
[ -f "$SOURCE_ENV" ] || { echo "missing baseline settings: $SOURCE_ENV" >&2; exit 1; }
[ ! -e "$ROOT/runs/carry_planner/$TAG" ] || { echo "existing run: $TAG" >&2; exit 3; }
set -a
. "$SOURCE_ENV"
set +a
export MS_CKPT="$stage1" MA_GPU="$GPU"
export CARRY_PLANNER_INIT="$ROOT/runs/carry_planner/carry_implicit_curvature05_lr3e4_clip20_3000_s0/planner_000600.pth"
[ -f "$CARRY_PLANNER_INIT" ] || { echo "missing checkpoint: $CARRY_PLANNER_INIT" >&2; exit 1; }
export CARRY_PLANNER_ANALYTIC_CURVATURE_COEF=0.5
export CARRY_PLANNER_LR=0.0003 CARRY_PLANNER_CLIP=0.2
export CARRY_PLANNER_ITERS=2400 CARRY_PLANNER_SEED=0 CARRY_PLANNER_ENVS=2048
export CARRY_PLANNER_OUTPUT="$ROOT/runs/carry_planner/$TAG"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$ROOT/runs/cache/cuda"
export TORCH_EXTENSIONS_DIR="$ROOT/runs/cache/torch_extensions"
export CONDA_BASE=${CONDA_BASE:-/home/hwanhee/anaconda3}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
. "$ROOT/mps/shell.sh"
mps_use "$GPU"
echo "tag=$TAG resume=600 iterations=601..3000 gpu=$GPU"
echo 'curvature=0.5 lr=0.0003 clip=0.2; remaining consistency7 settings preserved'
if [ "$MODE" = --dry-run ]; then
    env | LC_ALL=C sort | rg '^(CARRY_PLANNER_|CUDA_MPS_PIPE_DIRECTORY=|MS_CKPT=)'
    exit 0
fi
mkdir -p "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"
exec bash "$ROOT/TokenHSI-coord/carry_planner/train.sh" "$TAG" "$executor"
