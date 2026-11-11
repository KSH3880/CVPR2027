#!/bin/bash
# stage2_coupled_contact: eval
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_contact eval "$@"
