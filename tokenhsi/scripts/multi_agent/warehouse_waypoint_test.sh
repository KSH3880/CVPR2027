#!/bin/bash
set -eu
source "$(dirname "$0")/warehouse_waypoint_env.sh"
if [ "${HEADLESS:-0}" = 1 ]; then set -- "$@" --headless; fi
exec python tokenhsi/qualitative/warehouse_waypoint_viewer.py "$@"
