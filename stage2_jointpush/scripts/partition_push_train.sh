#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/partition_push_ca4_train.sh" "$@"
