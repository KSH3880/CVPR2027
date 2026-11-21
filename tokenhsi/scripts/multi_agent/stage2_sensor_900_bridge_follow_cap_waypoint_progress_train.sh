#!/bin/bash
set -eu
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
if [ -z "${RESUME_CHECKPOINT:-}" ]; then
  export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress.yaml}
fi
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_bridge_follow_cap_waypoint_progress train "$@"
