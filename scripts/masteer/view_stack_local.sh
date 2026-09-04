#!/bin/bash
# Frozen-ms18, no-training two-agent stacking probe.
#
#   bash scripts/masteer/view_stack_local.sh <tag-or-pth> [envs]
#
# Both agents move from the first step.  Agent 1 gets a low-speed steering
# command until the base box is released and stable; the controller then restores
# nominal speed and redirects its carry target/path to the measured box top.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
INPUT=${1:?usage: view_stack_local.sh <tag-or-pth> [envs]}
ENVS=${ENVS:-${2:-1}}

export MA_TOKEN=${MA_TOKEN:-mask}
export MS_TASK=HumanoidMAStackCarry
export MS_EVAL=1
export MA_SEP=0
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.5}
export MS_SCEN=free
export MS_MRAND=${MS_MRAND:-4}
export MS_CLIP=${MS_CLIP:-1}
export MS_LAT_MAX=${MS_LAT_MAX:-0}
export MS_CAM=${MS_CAM:-top}
export STACK_STABLE_STEPS=${STACK_STABLE_STEPS:-20}
export STACK_TOP_STABLE_STEPS=${STACK_TOP_STABLE_STEPS:-20}
export STACK_BASE_SCALE=${STACK_BASE_SCALE:-1.0}
export STACK_WAIT_SCALE=${STACK_WAIT_SCALE:-0.25}
export STACK_GO_SCALE=${STACK_GO_SCALE:-1.0}
export STACK_TOP_SIZE=${STACK_TOP_SIZE:-0.22}

exec bash "$ROOT/scripts/masteer/view_local.sh" "$INPUT" "$ENVS"
