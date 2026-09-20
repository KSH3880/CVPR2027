#!/bin/bash
# Holding k=10 experiment: one-command remote viewer.
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"

if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo "usage: TOKENHSI_GPU=5 bash $0 <checkpoint.pth> [agents=2] [envs=1] [objects=3] [repeats=10]"
    exit 0
fi
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
    echo "Provide an existing checkpoint.pth; paths are relative to the repository root." >&2
    echo "usage: TOKENHSI_GPU=5 bash $0 <checkpoint.pth> [agents] [envs] [objects] [repeats]" >&2
    exit 1
fi

export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output/approach_distance_success_holding_k10_vnc}
# run-gui discovers noVNC and websockify, defaults to port 6080, and shares
# TOKENHSI_GPU between CUDA and the viewer. All environment overrides still work.
exec bash "$SCRIPT_DIR/run-gui.sh" \
    bash "$SCRIPT_DIR/approach_distance_success_holding_k10_test.sh" \
    "$1" "${2:-2}" "${3:-1}" "${4:-3}" "${5:-10}"
