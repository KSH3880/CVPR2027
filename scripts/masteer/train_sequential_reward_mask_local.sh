#!/bin/bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms36_ms18init_seqrewardmask_s0}

export PILOT_INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
export PILOT_ENTRY_DELIVERED=1
export PILOT_ENTRY_LOWERED=1
export PILOT_ENTRY_STEPS=2
export PILOT_RELEASE_CARRY_BRIDGE=0
export PILOT_SEQUENTIAL_REWARD_MASK=1
export PILOT_CARRY_DONE_REWARD=1.6
export PILOT_RELEASE_DONE_REWARD=1.0
export PILOT_CLEAR_DONE_REWARD=1.5
export PILOT_ITERS=3000
export STACK_VIRTUAL_RETREAT_BOX=0

exec bash "$ROOT/scripts/masteer/train_clear_reward_pilot_local.sh" "$TAG"
