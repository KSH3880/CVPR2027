#!/bin/bash
# stage1: eval
exec bash "$(dirname "$0")/launch.sh" stage1 eval "$@"
