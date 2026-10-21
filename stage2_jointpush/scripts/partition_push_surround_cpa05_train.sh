#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/partition_push_surround_cpa05_training}
export PARTITION_PUSH_ENV_CONFIG=${PARTITION_PUSH_ENV_CONFIG:-$ROOT/ports/koo_push/tokenhsi/data/cfg/multi_agent/partition_push_surround_cpa05.yaml}
exec bash "$ROOT/scripts/partition_push_train.sh" "$@"
