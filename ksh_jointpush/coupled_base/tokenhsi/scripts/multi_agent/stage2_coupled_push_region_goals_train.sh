#!/bin/bash
# stage2_coupled_push_region_goals: train
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_push_region_goals train "$@"
