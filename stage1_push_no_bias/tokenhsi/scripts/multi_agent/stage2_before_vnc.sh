#!/bin/bash
# stage2_before: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_before vnc "$@"
