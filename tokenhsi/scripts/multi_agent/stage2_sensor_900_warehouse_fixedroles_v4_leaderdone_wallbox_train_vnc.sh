#!/bin/bash
set -eu
export ENV_CONFIG=${ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox.yaml}
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox train_vnc "$@"
