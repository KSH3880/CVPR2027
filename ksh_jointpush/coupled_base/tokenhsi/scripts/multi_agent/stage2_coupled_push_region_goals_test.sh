#!/bin/bash
# stage2_coupled_push_region_goals: eval
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_push_region_goals eval "$@"
