#!/bin/bash
set -eu
exec bash "$(dirname "$0")/launch.sh" stage1_push_no_bias vnc "$@"
