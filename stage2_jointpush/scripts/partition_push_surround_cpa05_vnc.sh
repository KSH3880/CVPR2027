#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/../runs/jointpush/partition_push_surround_cpa05_evaluation}
exec bash "$ROOT/scripts/partition_push_vnc.sh" "$@"
