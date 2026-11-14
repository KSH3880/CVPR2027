#!/bin/bash
# stage2_before_total_ft21k: train
exec bash "$(dirname "$0")/launch.sh" stage2_before_total_ft21k train "$@"
