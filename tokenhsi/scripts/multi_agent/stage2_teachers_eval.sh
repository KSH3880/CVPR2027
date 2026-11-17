#!/bin/bash
export HEADLESS=1 EVAL_PROTOCOL=general SEED=${SEED:-0}
if [ "$#" -lt 2 ]; then
  echo 'Usage: stage2_teachers_eval.sh <enable.pth> <coupled.pth> [agents=4] [envs=512] [objects=2*agents]' >&2
  exit 1
fi
exec bash "$(dirname "$0")/stage2_teachers_test.sh" "$1" "$2" "${3:-4}" "${4:-512}" "${5:-$((2*${3:-4}))}" 1
