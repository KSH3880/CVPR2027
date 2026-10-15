#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export JOINT_PUSH_REQUIRE_SURROUND60=1 RSI=loco
export PORT=${PORT:-6088}
exec bash "$ROOT/scripts/joint_push_aligned_test.sh" "$@"
