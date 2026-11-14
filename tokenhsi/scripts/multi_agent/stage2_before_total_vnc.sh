#!/bin/bash
# stage2_before_total: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_before_total vnc "$@"
