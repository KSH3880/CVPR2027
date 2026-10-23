#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PARTITION_PUSH_ENV_CONFIG=${PARTITION_PUSH_ENV_CONFIG:-tokenhsi/data/cfg/multi_agent/partition_push_task_only_ca4.yaml}
exec bash "$ROOT/scripts/partition_push_train.sh" "$@"
