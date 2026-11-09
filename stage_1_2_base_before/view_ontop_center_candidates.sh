#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
CASE=${1:-25}
if [ "$#" -gt 0 ]; then shift; fi
case "$CASE" in
  25) DIR=case_b1_e25_f0 ;;
  64) DIR=case_b1_e64_f2 ;;
  *) echo 'Usage: bash view_ontop_center_candidates.sh [25|64] [replay options]' >&2; exit 2 ;;
esac
exec bash "$ROOT/tokenhsi/scripts/multi_agent/replay_ontop_demo.sh" "$ROOT/output_etc/check/ontop_demo_center_full_seed50/$DIR" "$@"
