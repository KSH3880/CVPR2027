#!/bin/bash
set -eu
export HEADLESS=0
# Use an existing local display or launch this command inside your VNC desktop.
exec bash "$(dirname "$0")/partition_push_ca4_test.sh" "$@"
