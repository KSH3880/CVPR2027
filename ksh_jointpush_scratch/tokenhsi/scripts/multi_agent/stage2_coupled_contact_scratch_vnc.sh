#!/bin/bash
# stage2_coupled_contact_scratch: vnc
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_contact_scratch vnc "$@"
