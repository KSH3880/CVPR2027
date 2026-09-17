#!/bin/bash
# ms72: keep ms71's observable Base STACK hold reward, while applying the
# server-tested turn-away, forward retreat, endpoint deceleration, and strict
# endpoint/base-box/stop transition contract.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms72_ms18e9000_yawhybrid_endpoint12_hold1_anchor_nolegacy_1000_s0}

# Preserve ms71 STACK behavior exactly.
export PILOT_BASE_HOLD_REWARD_W=1.0
export PILOT_BASE_HOLD_ANCHOR_GATE=1
export PILOT_BASE_LEGACY_REWARD=0
export PILOT_BASE_REGRASP_PEN_W=0.0

# Server retreat controller: 2--3 m rear-biased path, forward walking at
# quarter scale, no around-box route, and a 12 cm endpoint-arrival latch.
export PILOT_RETREAT_SCALE=0.25
export PILOT_CLEAR_ROUTE_AROUND=0
export PILOT_RETREAT_ENDPOINT_TOL=0.12

# Linear yaw progress is active throughout the turn. Inside 90 degrees, add
# the alignment-dependent cosine gain to the same signed yaw progress.
export PILOT_CLEAR_YAW_PROGRESS=1
export PILOT_CLEAR_YAW_PROGRESS_W=1.5
export PILOT_CLEAR_HEADING_PROGRESS_W=2.0
export PILOT_CLEAR_YAW_COSINE_START_DEG=90.0

# Require the settled released box at the same time as endpoint arrival and
# the existing five-frame humanoid stop streak. Do not relax stop thresholds.
export PILOT_CLEAR_HARD_GATE=1
export PILOT_CLEAR_XY_TOL=0.10
export PILOT_CLEAR_Z_TOL=0.06
export PILOT_CLEAR_STABLE_LIN=0.08
export PILOT_CLEAR_STABLE_ANG=0.20

exec bash "$ROOT/scripts/masteer/train_retreat_hold_anchor_ms71_local.sh" "$TAG"
