#!/bin/bash
# ms71: preserve the complete ms70 CLEAR contract, then make Base's STACK
# reward observable: hold the retreat endpoint and remove hidden box-state and
# saturated CLEAR-completion credit.  Keep regrasp penalty at zero to isolate
# the endpoint-hold change.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms71_ms18e9000_yaw15_wait10_hold1_anchor_nolegacy_1000_s0}

export PILOT_BASE_HOLD_REWARD_W=${PILOT_BASE_HOLD_REWARD_W:-1.0}
export PILOT_BASE_HOLD_ANCHOR_GATE=1
export PILOT_BASE_LEGACY_REWARD=0
export PILOT_BASE_REGRASP_PEN_W=0.0

exec bash "$ROOT/scripts/masteer/train_retreat_turnaway_ms68_local.sh" "$TAG"
