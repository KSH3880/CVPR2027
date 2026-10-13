#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/joint_push_koo_rsi_ca4_test.sh" "$@"
