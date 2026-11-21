#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
export PORT=${PORT:-6081}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
source "$(dirname "$0")/runtime_env.sh"
exec bash tokenhsi/scripts/multi_agent/run-gui.sh \
  bash tokenhsi/scripts/multi_agent/bridge_wait_test.sh "$@"
