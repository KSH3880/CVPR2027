#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-0}
source "$(dirname "$0")/warehouse_waypoint_training_env.sh"
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop train "$@"
