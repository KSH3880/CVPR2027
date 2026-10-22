#!/bin/bash
set -eu
source "$(dirname "$0")/koo_1021_common.sh"
koo_checkpoint "${1:-$KOO_PROJECT/jointpush/checkpoints/koo_1021/Stage1BasePush_00007500.pth}" train
export KOO_PARTIAL_WORK=${KOO_PARTIAL_WORK:-$KOO_PROJECT/runs/jointpush/koo_partial_$(date +%Y%m%d_%H%M%S)/workspace}
if [ ! -f "$KOO_PARTIAL_WORK/configs/env.yaml" ]; then
 "$KOO_PYTHON" "$KOO_PROJECT/jointpush/scripts/koo_partial_prepare.py" --checkpoint "$CHECKPOINT" --workspace "$KOO_PARTIAL_WORK"
fi
exec bash "$KOO_PROJECT/jointpush/scripts/koo_partial_launch.sh" train
