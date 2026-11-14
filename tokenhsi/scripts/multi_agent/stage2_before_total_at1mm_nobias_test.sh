#!/bin/bash
# stage2_before_total_at1mm_nobias: eval
exec bash "$(dirname "$0")/launch.sh" stage2_before_total_at1mm_nobias eval "$@"
