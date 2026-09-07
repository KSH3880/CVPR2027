#!/bin/bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms34_ms18init_delivered_steer_s0}

export PILOT_INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
export STACK_ENTRY_DELIVERED=1
export STACK_ENTRY_STEPS=1
export STACK_VIRTUAL_RETREAT_BOX=0

exec bash "$ROOT/scripts/masteer/train_clear_reward_pilot_local.sh" "$TAG"