#!/bin/bash
set -eu
export PORT=${PORT:-6081}
export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
source "$(dirname "$0")/warehouse_waypoint_env.sh"
exec bash tokenhsi/scripts/multi_agent/run-gui.sh \
  bash tokenhsi/scripts/multi_agent/warehouse_waypoint_test.sh "$@"
