#!/bin/bash
# Record a trained steering policy to mp4, under the settings it TRAINED with.
#
#   bash scripts/steer/record.sh <ckpt> <src_tag> <port> <seconds> [gpu] [extra KEY=VAL ...]
#
# view.sh hardcodes legacy defaults (STEER_PHASEFREE=0, STEER_XTRACK=1.0,
# STEER_AIM=lookahead) and does NOT read the run's sidecar. Rendering a modern
# checkpoint under those defaults builds a 12-wide observation for a policy that
# expects 24 (STEER_DUAL), so it either crashes on a shape mismatch or shows a
# policy being fed the wrong thing. Always replay the sidecar first.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CKPT=$1; SRC=$2; PORT=$3; SECS=$4; GPU=${5:-6}; shift 5
python3 "$ROOT/scripts/vulkan_gpu_guard.py" "$GPU" >/dev/null || exit $?
set -a; . "$ROOT/runs/queue/logs/$SRC.env" 2>/dev/null || true; set +a
for kv in "$@"; do export "$kv"; done
export RECORD=1 RECORD_DELAY=${RECORD_DELAY:-70} PORT=$PORT STEER_GPU=$GPU CUDA_VISIBLE_DEVICES=$GPU
setsid bash "$ROOT/scripts/steer/view.sh" "$CKPT" "${ENVS_N:-3}" > "$ROOT/runs/videos/steer/log_$PORT.txt" 2>&1 &
VP=$!
sleep $((SECS + ${RECORD_DELAY:-70} + 20))
pkill -INT -g $(ps -o pgid= -p $VP | tr -d ' ') 2>/dev/null
sleep 8
pkill -g $(ps -o pgid= -p $VP | tr -d ' ') 2>/dev/null || true
