#!/bin/bash
# stage2_before_at1mm: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_before_at1mm vnc "$@"
