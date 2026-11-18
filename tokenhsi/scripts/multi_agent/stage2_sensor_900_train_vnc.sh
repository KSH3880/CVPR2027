#!/bin/bash
exec bash "$(dirname "$0")/launch.sh" stage2_sensor_900 train_vnc "$@"
