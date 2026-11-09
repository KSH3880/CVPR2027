#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
sample=${1:-1}
if (( $# )); then shift; fi
case "$sample" in
  1) case_dir=sit2_demo_walk_seed60/case_b1_e41_f0 ;;
  2) case_dir=sit2_demo_walk_seed62/case_b0_e58_f0 ;;
  *) echo "사용법: bash view_sit_samples.sh 1|2 [viewer 옵션]"; exit 2 ;;
esac
exec bash "$ROOT/tokenhsi/scripts/multi_agent/replay_ontop_demo.sh" "$ROOT/output_etc/check/$case_dir" "$@"
