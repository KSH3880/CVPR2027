#!/bin/bash
set -eu
source "$(dirname "$0")/partial_common.sh"
export PARTIAL_WORK=${PARTIAL_WORK:-$PARTIAL_ROOT/runs/jointpush/raw_cuboid_partial_$(date +%Y%m%d_%H%M%S)/workspace}
if [ ! -f "$PARTIAL_WORK/configs/env.yaml" ]; then
 args=(--workspace "$PARTIAL_WORK")
 if [ -n "${1:-}" ]; then args+=(--checkpoint "$1"); fi
 CUDA_VISIBLE_DEVICES='' "$PARTIAL_PYTHON" "$PARTIAL_ROOT/jointpush/scripts/partial_prepare.py" "${args[@]}"
else
 if [ -n "${1:-}" ]; then
  requested=$(realpath "$1")
  configured=$("$PARTIAL_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"]["transfer"]["checkpoint"])' "$PARTIAL_WORK/configs/env.yaml")
  [ "$requested" = "$configured" ] || { echo 'Workspace already pins a different baseline; use a new PARTIAL_WORK.' >&2; exit 1; }
 fi
fi
exec bash "$PARTIAL_ROOT/jointpush/scripts/partial_launch.sh" train
