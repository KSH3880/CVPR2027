#!/bin/bash
# stage2_coupled_contact: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_contact vnc "$@"
