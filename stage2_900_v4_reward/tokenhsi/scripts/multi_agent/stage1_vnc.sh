#!/bin/bash
# stage1: vnc
exec bash "$(dirname "$0")/launch.sh" stage1 vnc "$@"
