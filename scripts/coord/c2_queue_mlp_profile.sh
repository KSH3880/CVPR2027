#!/bin/bash
# Run the flat-MLP comparison only after the queued feature experiment is done.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WAIT_TAG=${1:-c2_k1_profile_features_s0}
TAG=${2:-c2_k1_mlp_profile_s0}
EXECUTOR=${3:-ms18_maskteam_origscale_c06_s0}

echo "C2_MLP_QUEUE_WAIT wait_tag=$WAIT_TAG next_tag=$TAG"
while pgrep -f "[c]2_queue_profile_features.sh .*${WAIT_TAG}" >/dev/null; do
    sleep 20
done
echo "C2_MLP_QUEUE_START tag=$TAG"
bash "$ROOT/scripts/coord/c2_mlp_profile_compare.sh" "$TAG" "$EXECUTOR" &
runner_pid=$!
while ! pgrep -f "[c]oordinator.train_closed_loop.*runs/coord/$TAG" >/dev/null; do
    if ! kill -0 "$runner_pid" 2>/dev/null; then wait "$runner_pid"; exit $?; fi
    sleep 2
done
bash "$ROOT/scripts/coord/c2_milestone_watch.sh" "$TAG" 50 100 250 &
watcher_pid=$!
wait "$runner_pid"; runner_status=$?
wait "$watcher_pid"; watcher_status=$?
echo "C2_MLP_QUEUE_DONE tag=$TAG runner_status=$runner_status watcher_status=$watcher_status"
if [ "$runner_status" -ne 0 ]; then exit "$runner_status"; fi
exit "$watcher_status"
