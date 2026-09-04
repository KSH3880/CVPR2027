#!/bin/bash
# GPU0: selected hard-gate + transition bonus method.
# GPU1: paired ablation with the one-shot transition bonus removed.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOGS="$ROOT/runs/queue/logs"
mkdir -p "$LOGS"

MAIN_TAG=${MAIN_TAG:-ms20_stackrel_gate_bonus3_s0}
ABL_TAG=${ABL_TAG:-ms20_stackrel_gate_bonus0_s0}

nohup setsid env MA_GPU=0 STACK_TRANSITION_BONUS=3.0 \
    bash "$ROOT/scripts/masteer/train_stack_release_local.sh" "$MAIN_TAG" \
    > "$LOGS/$MAIN_TAG.log" 2>&1 &
echo $! > "$LOGS/$MAIN_TAG.pid"

nohup setsid env MA_GPU=1 STACK_TRANSITION_BONUS=0.0 \
    bash "$ROOT/scripts/masteer/train_stack_release_local.sh" "$ABL_TAG" \
    > "$LOGS/$ABL_TAG.log" 2>&1 &
echo $! > "$LOGS/$ABL_TAG.pid"

echo "started $MAIN_TAG on GPU0 and $ABL_TAG on GPU1"
