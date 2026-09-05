#!/bin/bash
# Evaluate one retained stack checkpoint and print the first failed stage.
# Usage: stack_stage_eval.sh <tag> <gpu:6|7> <initial|epoch|latest> [envs]
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: stack_stage_eval.sh <tag> <gpu:6|7> <initial|epoch|latest> [envs]}
GPU=${2:?usage: stack_stage_eval.sh <tag> <gpu:6|7> <initial|epoch|latest> [envs]}
EPOCH=${3:-latest}
ENVS=${4:-512}
case "$GPU" in 6|7) ;; *) echo "GPU must be 6 or 7" >&2; exit 2 ;; esac

OUT=$ROOT/TokenHSI-masteer/output/masteer/$TAG
if [ "$EPOCH" = initial ]; then
    ENV_FILE=$ROOT/runs/queue/logs/$TAG.env
    if [ ! -f "$ENV_FILE" ]; then
        echo "sidecar not found: $ENV_FILE" >&2
        exit 4
    fi
    CK=$(bash -c 'source "$1"; printf "%s" "${MA_INIT_CKPT:-}"' _ "$ENV_FILE")
elif [ "$EPOCH" = latest ]; then
    CK=$(find "$OUT" -type f -name Humanoid.pth -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
else
    case "$EPOCH" in *[!0-9]*) echo "epoch must be initial, an integer, or latest" >&2; exit 2 ;; esac
    printf -v CK_NAME 'Humanoid_%08d.pth' "$EPOCH"
    CK=$(find "$OUT" -type f -name "$CK_NAME" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -z "${CK:-}" ]; then
    echo "checkpoint not found: tag=$TAG epoch=$EPOCH" >&2
    exit 4
fi

SUFFIX=${STACK_STAGE_SUFFIX:-stage_e$EPOCH}
QUEUE_EVAL_SOURCE=$TAG QUEUE_EVAL_CKPT=$CK MS_EVAL_SUFFIX=$SUFFIX \
TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo} \
bash "$ROOT/scripts/masteer/eval_one.sh" "$TAG" "$GPU" "$ENVS"

METRICS=$ROOT/runs/results/masteer/eval_${TAG}__${SUFFIX}.npy
BASE_METRICS=${STACK_STAGE_BASELINE:-$ROOT/runs/results/masteer/eval_${TAG}__stage_einitial.npy}
if [ "$EPOCH" != initial ] && [ -f "$BASE_METRICS" ]; then
    python3 "$ROOT/scripts/masteer/stack_stage_summary.py" "$METRICS" "$BASE_METRICS"
else
    python3 "$ROOT/scripts/masteer/stack_stage_summary.py" "$METRICS"
fi
