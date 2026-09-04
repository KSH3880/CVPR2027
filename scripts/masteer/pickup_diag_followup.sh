#!/bin/bash
# Wait for both detached 3K runs, evaluate, and continue both to 25K only when
# both policies miss the strict pickup gate.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG_DIR=$ROOT/runs/logs/masteer_pickup
STATUS=$LOG_DIR/pickupdiag_followup.status
ZERO=pickupdiag_zero_3k_s0
LIVE=pickupdiag_live_3k_s0

status() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "$STATUS"
}

status "waiting for 3K zero/live"
while [ ! -f "$LOG_DIR/$ZERO.exit" ] || [ ! -f "$LOG_DIR/$LIVE.exit" ]; do
    sleep 60
done

if ! grep -q '^rc=0 ' "$LOG_DIR/$ZERO.exit" || ! grep -q '^rc=0 ' "$LOG_DIR/$LIVE.exit"; then
    status "stopped: at least one 3K training failed"
    exit 1
fi

status "3K complete; starting strict evaluations"
PICKUP_EVAL_SUFFIX=gate3k bash "$ROOT/scripts/masteer/pickup_diag_eval.sh" "$ZERO" 6 512 \
    > "$LOG_DIR/launch_eval_${ZERO}_gate3k.log" 2>&1 &
P0=$!
PICKUP_EVAL_SUFFIX=gate3k bash "$ROOT/scripts/masteer/pickup_diag_eval.sh" "$LIVE" 7 512 \
    > "$LOG_DIR/launch_eval_${LIVE}_gate3k.log" 2>&1 &
P1=$!
wait "$P0"; R0=$?
wait "$P1"; R1=$?
if [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ]; then
    status "stopped: 3K evaluation failed zero=$R0 live=$R1"
    exit 1
fi

read_metric() {
    local tag=$1 key=$2
    awk -v key="$key" '
        /^PICKUP_DIAG / {
            for (i=1; i<=NF; i++) if ($i ~ ("^" key "=")) {
                split($i, a, "="); value=a[2]
            }
        }
        END { if (value != "") print value }
    ' "$LOG_DIR/eval_${tag}_gate3k.log"
}

ZP=$(read_metric "$ZERO" pick); ZB=$(read_metric "$ZERO" bothPick)
LP=$(read_metric "$LIVE" pick); LB=$(read_metric "$LIVE" bothPick)
if [ -z "$ZP" ] || [ -z "$ZB" ] || [ -z "$LP" ] || [ -z "$LB" ]; then
    status "stopped: missing 3K pickup metrics"
    exit 1
fi
status "3K gate zero pick=$ZP bothPick=$ZB; live pick=$LP bothPick=$LB"

both_fail=$(awk -v zp="$ZP" -v zb="$ZB" -v lp="$LP" -v lb="$LB" \
    'BEGIN { print ((zp < 0.85 || zb < 0.70) && (lp < 0.85 || lb < 0.70)) ? 1 : 0 }')
if [ "$both_fail" != 1 ]; then
    status "25K not started: at least one policy passed the strict gate"
    exit 0
fi

status "both missed gate; resuming each 3K checkpoint to total 25K"
bash "$ROOT/scripts/masteer/pickup_diag_train.sh" zero 25000 6 2048 \
    > "$LOG_DIR/launch_pickupdiag_zero_25k_s0.log" 2>&1 &
P0=$!
bash "$ROOT/scripts/masteer/pickup_diag_train.sh" live 25000 7 2048 \
    > "$LOG_DIR/launch_pickupdiag_live_25k_s0.log" 2>&1 &
P1=$!
wait "$P0"; R0=$?
printf 'rc=%s finished=%s\n' "$R0" "$(date '+%F %T %Z')" > "$LOG_DIR/pickupdiag_zero_25k_s0.exit"
wait "$P1"; R1=$?
printf 'rc=%s finished=%s\n' "$R1" "$(date '+%F %T %Z')" > "$LOG_DIR/pickupdiag_live_25k_s0.exit"
if [ "$R0" -ne 0 ] || [ "$R1" -ne 0 ]; then
    status "stopped: 25K training failed zero=$R0 live=$R1"
    exit 1
fi

status "25K complete; starting strict evaluations"
PICKUP_EVAL_SUFFIX=full25k bash "$ROOT/scripts/masteer/pickup_diag_eval.sh" pickupdiag_zero_25k_s0 6 512 \
    > "$LOG_DIR/launch_eval_pickupdiag_zero_25k_s0.log" 2>&1 &
P0=$!
PICKUP_EVAL_SUFFIX=full25k bash "$ROOT/scripts/masteer/pickup_diag_eval.sh" pickupdiag_live_25k_s0 7 512 \
    > "$LOG_DIR/launch_eval_pickupdiag_live_25k_s0.log" 2>&1 &
P1=$!
wait "$P0"; R0=$?
wait "$P1"; R1=$?
status "finished: 25K evaluations zero=$R0 live=$R1"
exit $((R0 != 0 || R1 != 0))
