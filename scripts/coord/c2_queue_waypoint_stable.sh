#!/bin/bash
# Preserve the first direct-waypoint run; start the stabilized variant when one GPU slot frees.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WAIT_TAG=${1:-c2_k1_profile_features_s0}
TAG=${2:-c2_k1_waypoint_stable_s0}
EXECUTOR=${3:-ms18_maskteam_origscale_c06_s0}

echo "C2_WAYPOINT_STABLE_WAIT wait_tag=$WAIT_TAG next_tag=$TAG"
while pgrep -f "[c]2_queue_profile_features.sh .*${WAIT_TAG}" >/dev/null; do
    sleep 20
done
echo "C2_WAYPOINT_STABLE_START tag=$TAG"
bash "$ROOT/scripts/coord/c2_queue_waypoint_mlp.sh" "$TAG" "$EXECUTOR"
