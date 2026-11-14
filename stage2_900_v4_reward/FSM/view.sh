#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-7}
GPU_INDEX=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
case "$GPU_INDEX" in 6|7) ;; *) echo 'FSM viewer: use TOKENHSI_GPU=6 or 7.' >&2; exit 1 ;; esac
source "$ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
exec python "$ROOT/FSM/run.py" "$@"
