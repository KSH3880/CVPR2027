#!/bin/bash
# Schema-4 interaction experiment: one-command remote viewer.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    echo "usage: TOKENHSI_GPU=5 TASK_GRAPH=hold_sit bash $0 <checkpoint.pth> [agents=2] [envs=1] [objects=3] [repeats=10]"
    exit 0
fi
if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
    echo "Provide an existing checkpoint.pth; paths are relative to the repository root." >&2
    exit 1
fi
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export OUTPUT_PATH=${OUTPUT_PATH:-output/approach_distance_edge_context_interaction_vnc}
export TASK_GRAPH=${TASK_GRAPH:-hold_sit}
exec bash "$SCRIPT_DIR/run-gui.sh" \
    bash "$SCRIPT_DIR/approach_distance_edge_context_interaction_test.sh" \
    "$1" "${2:-2}" "${3:-1}" "${4:-3}" "${5:-10}"
