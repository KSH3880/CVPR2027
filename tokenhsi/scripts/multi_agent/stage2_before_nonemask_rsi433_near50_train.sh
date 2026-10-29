#!/bin/bash
# stage2_before_nonemask_rsi433_near50: train
exec bash "$(dirname "$0")/launch.sh" stage2_before_nonemask_rsi433_near50 train "$@"
