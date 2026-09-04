#!/bin/bash
# GPU 0: agent 0 legacy carry, agent 1 release/clear onto a thin shelf.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms19_release_shelf_c06_s0}

export MA_GPU=0
export REL_SUPPORT_MODE=shelf
exec bash "$ROOT/scripts/masteer/train_dual_release_local.sh" "$TAG"
