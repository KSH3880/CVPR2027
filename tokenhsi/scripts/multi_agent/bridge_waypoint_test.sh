#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
source "$(dirname "$0")/runtime_env.sh"
exec python tokenhsi/qualitative/bridge_waypoint_viewer.py "$@"
