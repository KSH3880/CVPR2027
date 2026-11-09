#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
exec bash "$ROOT/tokenhsi/scripts/multi_agent/replay_ontop_demo.sh" "$ROOT/output_etc/check/ontop_demo_extent_carry_seed49/case_b7_e88_f2" "$@"
