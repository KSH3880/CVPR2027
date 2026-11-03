#!/bin/bash
# stage2_before_rsi433: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_before_rsi433 vnc "$@"
