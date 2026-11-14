#!/bin/bash
exec bash "$(dirname "$0")/launch.sh" stage2 eval "$@"
