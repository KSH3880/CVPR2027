#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export JOINT_PUSH_REQUIRE_RANDOM_SPAWN60=1
export RSI=loco
export PORT=${PORT:-6087}
exec bash "$ROOT/scripts/joint_push_aligned_test.sh" "$@"
