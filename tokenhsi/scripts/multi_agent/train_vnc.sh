#!/bin/bash
# <experiment> [agents] [envs=4] [objects]
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../../.."
EXPERIMENT=${1:?usage: train_vnc.sh <experiment> [agents] [envs=4] [objects]}
case "$EXPERIMENT" in
  stage1_jointpush) AGENTS=2; OBJECTS=4 ;;
  stage1|stage1_v2|stage1_v2_fixed|stage1_v2_fixed_push|stage1_v2_fixed_push_360|stage1_v2_fixed_push_360_raw|stage1_v2_fixed_push_360_raw_mass20to25|stage1_v2_fixed_push_360_raw_cuboid|stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform|stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130) AGENTS=2; OBJECTS=4 ;;
  stage2) AGENTS=4; OBJECTS=8 ;;
  stage2_shared_full_success|stage2_shared_region_success) AGENTS=4; OBJECTS=12 ;;
  *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 1 ;;
esac
AGENTS=${2:-$AGENTS}
ENVS=${3:-4}
export PORT=${PORT:-6080}
export HEADLESS=0
export TASK_CAMERA=${TASK_CAMERA:-overview}
export MINIBATCH_SIZE=${MINIBATCH_SIZE:-$((ENVS * AGENTS * 32))}
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/$EXPERIMENT/train_vnc}
exec bash "$SCRIPT_DIR/run-gui.sh" bash "$SCRIPT_DIR/${EXPERIMENT}_train.sh" \
  "$AGENTS" "$ENVS" "${4:-$OBJECTS}"
