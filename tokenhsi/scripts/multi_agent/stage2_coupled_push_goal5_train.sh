#!/bin/bash
# stage2_coupled_push_goal5: train
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_push_goal5 train "$@"
