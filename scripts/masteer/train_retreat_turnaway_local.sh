#!/bin/bash
# Turn and walk forward along the object-free CLEAR steering path, then restore
# ms52's Top-wait and Base-STACK hold rewards. Carry handling stays zero-fade only.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms66_ms18e9000_timefade10_heading05_classicsteer_3000_s0}

export PILOT_GPU=1
export PILOT_NEGATIVE_CLEAR_REWARD=0
export PILOT_CLEAR_CLASSIC_STEER=1
export PILOT_CLEAR_HEADING_PROGRESS_W=0.5
export PILOT_CARRY_OBS_ZERO_FADE_STEPS=10
export PILOT_CLEAR_MOVE_W=0.0
export PILOT_CLEAR_FORWARD_GATE=0
export PILOT_RETREAT_DIST=2.0
export PILOT_RETREAT_DIST_MIN=1.2
export PILOT_STOP_DECEL_DIST=0.40
export PILOT_STOP_HOLD_STEPS=5
export PILOT_STOP_LIN=0.20
export PILOT_STOP_ANG=1.00
export PILOT_STOP_UPRIGHT_DEG=25.0
export PILOT_STOP_CONTACT_FORCE=0.50
export PILOT_TOP_WAIT_REWARD_W=0.10
export PILOT_BASE_HOLD_REWARD_W=0.10
export PILOT_BASE_REGRASP_PEN_W=0.0

exec bash "$ROOT/scripts/masteer/train_retreat_multisteer_mask_local.sh" "$TAG"
