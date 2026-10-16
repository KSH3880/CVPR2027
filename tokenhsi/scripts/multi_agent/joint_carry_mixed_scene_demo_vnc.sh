#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
if [ "${DEMO_AGENTS:-6}" = 4 ]; then
    export OUTPUT_PATH=${OUTPUT_PATH:-output/joint_carry_mixed_scene_demo_4agents}
else
    export OUTPUT_PATH=${OUTPUT_PATH:-output/joint_carry_mixed_scene_demo}
fi
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/joint_carry_mixed_scene_demo_test.sh" "$@"
