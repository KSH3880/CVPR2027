#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export PORT=${PORT:-6082} VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
export HEADLESS=0
source "$(dirname "$0")/runtime_env.sh"
exec bash tokenhsi/scripts/multi_agent/run-gui.sh \
  bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_zeroshot_test.sh "$@"
