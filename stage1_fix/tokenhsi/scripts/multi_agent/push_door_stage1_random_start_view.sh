#!/bin/bash
# Local desktop viewer; uses the existing display, without VNC.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 <checkpoint.pth> [num_envs=1] [repeats=10]" >&2
    exit 1
fi
if [ -z "${DISPLAY:-}" ]; then
    echo 'No local DISPLAY. Run this script in a desktop terminal.' >&2
    exit 1
fi
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export HEADLESS=0
export OUTPUT_PATH=${OUTPUT_PATH:-output/push_door_stage1_random_start_local_viewer}
echo '[viewer] GPU VRAM before launch (MiB):'
nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
exec bash "$SCRIPT_DIR/push_door_stage1_random_start_test.sh" "$1" "${2:-1}" "${3:-10}"
