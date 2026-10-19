#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
. "$SCRIPT_DIR/gui_gpu_env.sh"
[ "$TOKENHSI_GUI_GPU_INDEX" = 6 ] || { echo 'Only physical GPU 6 allowed' >&2; exit 1; }
. "$SCRIPT_DIR/runtime_env.sh"
exec python tokenhsi/scripts/multi_agent/lift_support_pose.py "$@"
