#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ $# -gt 1 ]; then
    echo "usage: $0 [CPA-Release1-checkpoint.pth]" >&2
    exit 1
fi
CKPT=${1:-output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1TaskEmbedding_12-20-50-28/nn/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1TaskEmbedding_00019000.pth}
export DEMO_CA_ENTMAX=1
export OUTPUT_PATH=${OUTPUT_PATH:-output/joint_carry_mixed_scene_demo_ca_entmax}
exec bash "$SCRIPT_DIR/joint_carry_mixed_scene_demo_test.sh" "$CKPT"
