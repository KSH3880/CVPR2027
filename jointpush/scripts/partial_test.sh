#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export CHECKPOINT=${1:?Pass Stage2 checkpoint or RawCuboid baseline checkpoint}
export PARTIAL_WORK=${PARTIAL_WORK:?Pass the training workspace with its exact configs}
export HEADLESS=${HEADLESS:-1} NUM_ENVS=${NUM_ENVS:-1}
exec bash "$ROOT/jointpush/scripts/partial_launch.sh" test
