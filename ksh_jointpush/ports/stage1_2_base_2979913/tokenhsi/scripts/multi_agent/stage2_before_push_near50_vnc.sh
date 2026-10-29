#!/bin/bash
# stage2_before_push_near50: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_before_push_near50 vnc "$@"
