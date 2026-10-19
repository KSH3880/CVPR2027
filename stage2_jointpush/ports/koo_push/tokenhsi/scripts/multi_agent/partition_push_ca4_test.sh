#!/bin/bash
set -eu
export PARTITION_PUSH_MODE=test
export PARTITION_PUSH_CHECKPOINT=${1:?Provide a PartitionPush checkpoint}
export NUM_ENVS=${NUM_ENVS:-16}
exec bash "$(dirname "$0")/partition_push_ca4_train.sh"
