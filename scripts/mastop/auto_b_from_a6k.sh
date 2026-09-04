#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
A_PID=${A_PID:?A_PID is required}
A_TAG=${A_TAG:-pickupa_minimal_clip1_oldcarry_15k_s0}
B_TAG=${B_TAG:-mastop_b_from_a6k_s0}
A_OUT=$ROOT/runs/output/masteer_pickup/$A_TAG
LOG_DIR=$ROOT/runs/logs/mastop_carry
CLAIM=$LOG_DIR/.${B_TAG}.watcher
INTERVAL=${WATCH_INTERVAL:-60}

mkdir -p "$LOG_DIR"
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "watcher already active: $CLAIM" >&2
    exit 3
fi
cleanup() { rmdir "$CLAIM" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[$(date '+%F %T %Z')] waiting A_PID=$A_PID A_TAG=$A_TAG checkpoint=6000"
while true; do
    CKPT=$(find "$A_OUT" -name Humanoid_00006000.pth -type f -print -quit 2>/dev/null || true)
    if [ -n "$CKPT" ]; then
        SIZE1=$(stat -c %s "$CKPT")
        sleep 10
        SIZE2=$(stat -c %s "$CKPT")
        if [ "$SIZE1" -gt 0 ] && [ "$SIZE1" -eq "$SIZE2" ]; then
            if /home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python - "$CKPT" <<'PYCHECK'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
if int(checkpoint.get("epoch", 0)) < 6000:
    raise RuntimeError("checkpoint epoch is below 6000")
PYCHECK
            then
                break
            fi
        fi
    fi
    if ! kill -0 "$A_PID" 2>/dev/null; then
        echo "[$(date '+%F %T %Z')] A exited before a valid 6k checkpoint" >&2
        exit 4
    fi
    sleep "$INTERVAL"
done

echo "[$(date '+%F %T %Z')] valid checkpoint=$CKPT; launching B on GPU 6"
MSTOP_START_CKPT="$CKPT" \
MSTOP_TAG="$B_TAG" \
MSTOP_GPU=6 \
MSTOP_ENVS=2048 \
MSTOP_ITERS=3000 \
MSTOP_CONDA_ENV=tokenhsi_koo \
bash "$ROOT/TokenHSI-mastop/tokenhsi/scripts/tokenhsi/stage2_ma_stop_residual_train.sh"
