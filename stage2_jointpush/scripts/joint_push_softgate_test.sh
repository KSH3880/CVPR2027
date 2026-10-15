#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PORT=${PORT:-6089} JOINT_PUSH_REQUIRE_SOFTGATE=1
exec bash "$ROOT/scripts/joint_push_surround_test.sh" "$@"
