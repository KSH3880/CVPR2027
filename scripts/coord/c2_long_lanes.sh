#!/bin/bash
# Three bounded GPU lanes: weighted->original w3, weighted->original free, explicit resume.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
EXECUTOR=${1:-ms18_maskteam_origscale_c06_s0}

bash "$ROOT/scripts/coord/c2_slowwin_to3000.sh" \
    c2_k1_slowwin_w3_timebias3_s0 3 - 3 0 "$EXECUTOR" && \
bash "$ROOT/scripts/coord/c2_slowwin_to3000.sh" \
    c2_k1_slowwin_w3_delay_s0 3 \
    "$ROOT/runs/coord/c2_k1_slowwin_w3_delay_s0/coord_c2_000250.pth" \
    0 0 "$EXECUTOR" &
lane_w3=$!

bash "$ROOT/scripts/coord/c2_slowwin_to3000.sh" \
    c2_k1_slowwin_free_timebias3_s0 0 - 3 0 "$EXECUTOR" && \
bash "$ROOT/scripts/coord/c2_slowwin_to3000.sh" \
    c2_k1_slowwin_free_delay_s0 0 \
    "$ROOT/runs/coord/c2_k1_slowwin_free_delay_s0/coord_c2_000250.pth" \
    0 0 "$EXECUTOR" &
lane_free=$!

while ! rg -q '^C2_SLOWWIN_DONE tag=c2_k1_slowwin_explicit_s0 ' \
    "$ROOT/runs/queue/logs/c2_k1_slowwin_explicit_s0.log" 2>/dev/null; do
    sleep 20
done
bash "$ROOT/scripts/coord/c2_slowwin_to3000.sh" \
    c2_k1_slowwin_explicit_s0 0 \
    "$ROOT/runs/coord/c2_k1_slowwin_explicit_s0/coord_c2_000250.pth" \
    0 1 "$EXECUTOR" &
lane_explicit=$!

wait "$lane_w3"; status_w3=$?
wait "$lane_free"; status_free=$?
wait "$lane_explicit"; status_explicit=$?
echo "C2_LONG_LANES_DONE w3=$status_w3 free=$status_free explicit=$status_explicit"
if [ "$status_w3" -ne 0 ]; then exit "$status_w3"; fi
if [ "$status_free" -ne 0 ]; then exit "$status_free"; fi
exit "$status_explicit"
