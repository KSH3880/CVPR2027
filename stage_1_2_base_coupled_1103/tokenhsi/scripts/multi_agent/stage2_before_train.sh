#!/bin/bash
# stage2_before: train
exec bash "$(dirname "$0")/launch.sh" stage2_before train "$@"
