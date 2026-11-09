#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
exec bash "$ROOT/tokenhsi/scripts/multi_agent/replay_ontop_demo.sh" "$ROOT/output_etc/check/ontop_demo_walk_seed46/case_b0_e44_f3" "$@"
