#!/bin/bash
# Launch measured-timing C2 and evaluate milestones without stopping training.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-c2_k1_jointpoint_future8_meastime_s0}
EXECUTOR=${2:-ms18_maskteam_origscale_c06_s0}

bash "$ROOT/scripts/coord/c2_jointpoint_future8_meastime_compare.sh" \
    "$TAG" "$EXECUTOR" &
runner_pid=$!
while ! pgrep -f "[c]oordinator.train_closed_loop.*runs/coord/$TAG" >/dev/null; do
    if ! kill -0 "$runner_pid" 2>/dev/null; then wait "$runner_pid"; exit $?; fi
    sleep 2
done
bash "$ROOT/scripts/coord/c2_milestone_watch.sh" "$TAG" 20 50 100 250 &
watcher_pid=$!
wait "$runner_pid"; runner_status=$?
wait "$watcher_pid"; watcher_status=$?
echo "C2_FUTURE8_MEASTIME_DONE tag=$TAG runner_status=$runner_status watcher_status=$watcher_status"
if [ "$runner_status" -ne 0 ]; then exit "$runner_status"; fi
exit "$watcher_status"
