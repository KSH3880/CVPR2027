#!/bin/bash
set -eu
exec bash "$(dirname "$0")/launch.sh" stage1_push_schedule vnc "$@"
