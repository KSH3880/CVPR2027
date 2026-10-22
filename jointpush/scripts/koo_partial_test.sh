#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export CHECKPOINT=${1:?Provide the Partial Stage2 checkpoint or its Stage1 base for an identity policy}
export KOO_PARTIAL_WORK=${KOO_PARTIAL_WORK:?Use the training workspace with its exact configs}
export HEADLESS=${HEADLESS:-1} NUM_ENVS=${NUM_ENVS:-1}
exec bash "$ROOT/jointpush/scripts/koo_partial_launch.sh" test
