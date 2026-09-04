#!/bin/bash
# GPU 1: agent 0 legacy carry, agent 1 release/clear onto a solid box.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:-ms19_release_box_c06_s0}

export MA_GPU=1
export REL_SUPPORT_MODE=box
exec bash "$ROOT/scripts/masteer/train_dual_release_local.sh" "$TAG"
