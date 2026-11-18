#!/usr/bin/env bash
set -eu
unset CHECK MAX_ITERATIONS OUTPUT_PATH RESUME_CHECKPOINT ENV_CONFIG STAGE1_CHECKPOINT
exec bash "$(dirname "$0")/tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_train.sh" "$@"
