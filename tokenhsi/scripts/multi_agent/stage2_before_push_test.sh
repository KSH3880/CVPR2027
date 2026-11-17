#!/bin/bash
# stage2_before_push: eval
exec bash "$(dirname "$0")/launch.sh" stage2_before_push eval "$@"
