#!/bin/bash
set -eu
export TOKENHSI_GPU=6
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export WEBSOCKIFY=${WEBSOCKIFY:-/home/hwanhee/anaconda3/envs/tokenhsi/bin/websockify}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0 NUM_ENVS=${NUM_ENVS:-1} REPEATS=${REPEATS:-10000} PORT=${PORT:-6083}
exec bash "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/run-gui.sh" bash "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/joint_push_virtual_half_test.sh"
