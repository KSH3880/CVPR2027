#!/bin/bash
# stage2_coupled_push_individual_goal: train
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_push_individual_goal train "$@"
