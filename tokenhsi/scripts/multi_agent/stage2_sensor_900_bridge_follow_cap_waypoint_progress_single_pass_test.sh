#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass eval "$@"
