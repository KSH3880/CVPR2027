#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
exec bash "$ROOT/tokenhsi/scripts/multi_agent/replay_ontop_demo.sh" "$ROOT/output_etc/check/random8_demo_near_seed44/case_b2_e27" "$@"
