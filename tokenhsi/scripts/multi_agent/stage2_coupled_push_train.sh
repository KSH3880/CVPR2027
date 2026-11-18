#!/bin/bash
# stage2_coupled_push: train
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_push train "$@"
