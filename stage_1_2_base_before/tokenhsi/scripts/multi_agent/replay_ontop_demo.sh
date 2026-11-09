#!/bin/bash
# Exact recorded-state visualization; no policy inference or physics stepping.
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-}
export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-}
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
. "$SCRIPT_DIR/runtime_env.sh"
. "$SCRIPT_DIR/gui_gpu_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python tokenhsi/replay_ontop_demo.py "$@"
