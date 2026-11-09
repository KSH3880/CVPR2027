#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
exec bash "$ROOT/tokenhsi/scripts/multi_agent/replay_ontop_demo.sh" "$ROOT/output_etc/check/sit2_demo_walk_seed60/case_b1_e41_f0" "$@"
