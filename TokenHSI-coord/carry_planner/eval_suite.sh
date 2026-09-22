#!/bin/bash
# Run varied deterministic episode distributions over several seeds.
# Usage: eval_suite.sh <planner.pth> <frozen-ms18.pth> [envs] [seeds...]
set -eo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLANNER=${1:?usage: eval_suite.sh <planner.pth> <frozen-ms18.pth> [envs] [seeds...]}
POLICY=${2:?usage: eval_suite.sh <planner.pth> <frozen-ms18.pth> [envs] [seeds...]}
ENVS=${3:-64}
if [ "$#" -ge 3 ]; then shift 3; else set --; fi
if [ "$#" -eq 0 ]; then set -- 0 1 2; fi

for seed in "$@"; do
    for profile in mixed converge cross free; do
        bash "$HERE/eval_one.sh" "$PLANNER" "$POLICY" "$profile" "$ENVS" "$seed"
    done
done
