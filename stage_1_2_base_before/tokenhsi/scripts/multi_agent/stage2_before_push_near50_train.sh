#!/bin/bash
# stage2_before_push_near50: train
exec bash "$(dirname "$0")/launch.sh" stage2_before_push_near50 train "$@"
