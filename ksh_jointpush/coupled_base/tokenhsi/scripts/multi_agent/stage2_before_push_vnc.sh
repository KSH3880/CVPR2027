#!/bin/bash
# stage2_before_push: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_before_push vnc "$@"
