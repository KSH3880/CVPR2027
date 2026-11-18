#!/bin/bash
set -eu
if [ "$#" -lt 2 ] || [ ! -f "$1" ] || [ ! -f "$2" ]; then
  echo 'Usage: stage2_teachers_test.sh <enable.pth> <coupled.pth> [agents=4] [envs=1] [objects=2*agents] [repeats=10]' >&2
  exit 1
fi
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
case "$TOKENHSI_GPU" in 5|6) ;; *) echo 'Use TOKENHSI_GPU=5 or 6.' >&2; exit 1 ;; esac
export TEACHER_BEFORE=$1 TEACHER_COUPLED=$2
shift 2
PEOPLE=${1:-4}
if [ "$PEOPLE" -gt 4 ]; then export GROUP_SPAWN_OVER4=${GROUP_SPAWN_OVER4:-1}; fi
exec bash "$(dirname "$0")/launch.sh" stage2 "${TEACHER_VIEW_MODE:-eval}" Base \
  "$PEOPLE" "${2:-1}" "${3:-$((2*PEOPLE))}" "${4:-10}"
