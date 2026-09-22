#!/bin/bash
# MS79 ablation: restore the MS75 wait behavior and keep the released base
# agent carry-box observation zero throughout STACK and SUCCESS.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_TAG=ms75_ms18e9000_endcell18_hold1_regrasp1_hybridyaw_3000_s0
BASE_ENV="$ROOT/runs/queue/logs/$BASE_TAG.env"
TAG=${1:-ms80_ms75_faceparallel10_w025_nobaseboxobs_3000_s0}
GPU=${MA_GPU:-6}

if [[ "$GPU" != 6 && "$GPU" != 7 ]]; then
    echo "MA_GPU must be 6 or 7: $GPU" >&2
    exit 2
fi
if [ ! -f "$BASE_ENV" ]; then
    echo "ms75 sidecar 없음: $BASE_ENV" >&2
    exit 1
fi

source "$BASE_ENV"
export MS_TAG="$TAG"
export MA_GPU="$GPU"
export MS_ENVS=${PILOT_ENVS:-1024}
export MS_ITERS=${PILOT_ITERS:-3000}

export STACK_TOP_WAIT_XY_ZERO=0
export STACK_CARRY_OBS_ZERO_ON_STACK=1
export STACK_CARRY_OBS_KEEP_GOAL=0

# Final success: contacting faces within 10 degrees for the existing 20-frame
# stable-placement streak. Dense shaping is active only near the stack target.
export STACK_TOP_SURFACE_PARALLEL=1
export STACK_TOP_PARALLEL_DEG=10.0
export STACK_TOP_PARALLEL_REWARD_W=${PILOT_PARALLEL_W:-0.25}
export STACK_TOP_PARALLEL_REWARD_SIGMA_DEG=${PILOT_PARALLEL_SIGMA_DEG:-10.0}
export STACK_TOP_PARALLEL_REWARD_XY=${PILOT_PARALLEL_XY:-0.25}
export STACK_TOP_PARALLEL_REWARD_Z=${PILOT_PARALLEL_Z:-0.15}

# Hand use and release are not objectives in this pilot.
export STACK_TOP_REQUIRE_HAND_CLEAR=0
export STACK_TOP_RELEASE_PROGRESS_W=0
export STACK_TOP_HOLD_PEN_W=0
export STACK_TOP_SETTLE_STEPS=0

exec bash "$ROOT/scripts/masteer/train_local.sh" "$TAG"
