#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
A_PID=${A_PID:?A_PID is required}
A_TAG=${A_TAG:-pickupa_minimal_clip1_oldcarry_15k_s0}
BASELINE_TAG=${BASELINE_TAG:-mastop_b_from_a6k_s0}
B_TAG=${B_TAG:-mastop_b_phase_noamp_from_a6k_s0}
BASELINE_LOG=$ROOT/runs/logs/mastop_carry/auto_b_from_a6k_s0.watcher.log
A_OUT=$ROOT/runs/output/masteer_pickup/$A_TAG
LOG_DIR=$ROOT/runs/logs/mastop_carry
CLAIM=$LOG_DIR/.${B_TAG}.watcher
INTERVAL=${WATCH_INTERVAL:-5}

mkdir -p "$LOG_DIR"
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "watcher already active: $CLAIM" >&2
    exit 3
fi
printf '%s\n' "$$" > "$CLAIM/pid"
cleanup() { rm -f "$CLAIM/pid"; rmdir "$CLAIM" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[$(date '+%F %T %Z')] waiting baseline=$BASELINE_TAG A_TAG=$A_TAG exact checkpoint=6000"
while true; do
    LINE=$(grep "valid checkpoint=.*; launching B" "$BASELINE_LOG" 2>/dev/null | tail -1 || true)
    if [ -n "$LINE" ]; then
        CKPT=${LINE#*valid checkpoint=}
        CKPT=${CKPT%; launching B*}
        case "$CKPT" in "$A_OUT"/*/nn/Humanoid_00006000.pth) ;; *) CKPT= ;; esac
        if [ -n "$CKPT" ] && [ -s "$CKPT" ]; then
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
        echo "[$(date '+%F %T %Z')] A exited before baseline launched from a valid 6k checkpoint" >&2
        exit 4
    fi
    sleep "$INTERVAL"
done

echo "[$(date '+%F %T %Z')] baseline checkpoint=$CKPT; launching phase-noamp B on GPU 6"
MSTOP_START_CKPT="$CKPT" \
MSTOP_TAG="$B_TAG" \
MSTOP_GPU=6 \
MSTOP_ENVS=2048 \
MSTOP_ITERS=3000 \
MSTOP_SEED=0 \
MSTOP_PHASE_REWARD=1 \
MSTOP_CONDA_ENV=tokenhsi_koo \
bash "$ROOT/TokenHSI-mastop/tokenhsi/scripts/tokenhsi/stage2_ma_stop_residual_train.sh"
