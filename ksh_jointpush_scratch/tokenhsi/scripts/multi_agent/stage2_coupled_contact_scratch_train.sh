#!/bin/bash
# stage2_coupled_contact_scratch: train
exec bash "$(dirname "$0")/launch.sh" stage2_coupled_contact_scratch train "$@"
