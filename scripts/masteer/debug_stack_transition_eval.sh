#!/bin/bash
# Evaluation-only diagnosis of CLEAR->STACK transition. No training is started.
# Usage: debug_stack_transition_eval.sh <tag> <gpu:6|7> [initial|epoch|latest] [envs]
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: debug_stack_transition_eval.sh <tag> <gpu:6|7> [epoch] [envs]}
GPU=${2:?usage: debug_stack_transition_eval.sh <tag> <gpu:6|7> [epoch] [envs]}
EPOCH=${3:-latest}
ENVS=${4:-256}
CASES=${STACK_DEBUG_CASES:-"baseline base_carry_live base_continue top_fast top_keep_wait top_balanced candidate base_long top_wait_base_long top_move_base_long delayed_stack"}
TRACE_EVENTS=${STACK_TRACE_EVENTS:-16}
TRACE_STEPS=${STACK_TRACE_STEPS:-60}
RESULT_DIR="$ROOT/runs/results/masteer"
REPORT="$RESULT_DIR/debug_stack_${TAG}_e${EPOCH}.log"
mkdir -p "$RESULT_DIR"
touch "$REPORT"
echo "===== run $(date '+%F %T') cases=[$CASES] =====" | tee -a "$REPORT"

case "$GPU" in 6|7) ;; *) echo "GPU must be 6 or 7" >&2; exit 2 ;; esac

run_case() {
    local name=$1 override=$2
    local suffix="debug_${name}_e${EPOCH}"
    local trace="$RESULT_DIR/trace_${TAG}__${suffix}.npz"
    echo "===== $name override=[$override] =====" | tee -a "$REPORT"
    STACK_STAGE_SUFFIX="$suffix" \
    STACK_TRACE="$trace" STACK_TRACE_EVENTS="$TRACE_EVENTS" \
    STACK_TRACE_STEPS="$TRACE_STEPS" MS_EVAL_OVERRIDE="$override" \
        bash "$ROOT/scripts/masteer/stack_stage_eval.sh" \
            "$TAG" "$GPU" "$EPOCH" "$ENVS" | tee -a "$REPORT"
    if [ -f "$trace" ]; then
        python3 "$ROOT/scripts/masteer/stack_transition_trace_summary.py" \
            "$trace" | tee -a "$REPORT"
    else
        echo "STACK_TRACE_EMPTY case=$name (no STACK transition)" | tee -a "$REPORT"
    fi
}

for name in $CASES; do
    case "$name" in
        baseline) override="" ;;
        base_carry_live) override="STACK_DEBUG_CARRY_LIVE_ON_STACK=1" ;;
        base_continue) override="STACK_CLEAR_STOP_ON_STACK=0" ;;
        top_fast) override="STACK_TOP_SCALE=1.0" ;;
        top_keep_wait) override="STACK_DEBUG_KEEP_WAIT=1" ;;
        top_balanced) override="STACK_DEBUG_REQUIRE_TOP_BALANCED=1" ;;
        base_long)
            override="STACK_CLEAR_STOP_ON_STACK=0 STACK_RETREAT_DIST=2.0"
            ;;
        top_wait_base_long)
            override="STACK_CLEAR_STOP_ON_STACK=0 STACK_RETREAT_DIST=2.0 STACK_DEBUG_KEEP_WAIT=1"
            ;;
        top_move_base_long)
            override="STACK_CLEAR_STOP_ON_STACK=0 STACK_RETREAT_DIST=2.0 STACK_TOP_SCALE=1.0"
            ;;
        delayed_stack)
            override="STACK_CLEAR_STOP_ON_STACK=0 STACK_RETREAT_DIST=2.0 STACK_CLEAR_ARC_DIST=1.4 STACK_DEBUG_KEEP_WAIT=1"
            ;;
        candidate)
            override="STACK_DEBUG_CARRY_LIVE_ON_STACK=1 STACK_TOP_SCALE=1.0 STACK_DEBUG_REQUIRE_TOP_BALANCED=1"
            ;;
        *) echo "unknown STACK_DEBUG_CASE: $name" >&2; exit 2 ;;
    esac
    run_case "$name" "$override"
done

echo "STACK_DEBUG_REPORT $REPORT"
