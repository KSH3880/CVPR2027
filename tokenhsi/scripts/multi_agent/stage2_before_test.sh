#!/bin/bash
# stage2_before: eval
exec bash "$(dirname "$0")/launch.sh" stage2_before eval "$@"
