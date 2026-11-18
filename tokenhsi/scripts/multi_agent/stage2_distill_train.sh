#!/bin/bash
exec bash "$(dirname "$0")/launch.sh" stage2_distill train "$@"
