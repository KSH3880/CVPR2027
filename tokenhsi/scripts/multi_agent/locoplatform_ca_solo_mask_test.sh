#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -gt 1 ]; then
    echo "usage: $0 [LocoPlatform-checkpoint.pth]" >&2
    exit 1
fi
CKPT=${1:-output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1LocoPlatformTaskEmbedding_13-16-32-42/nn/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1LocoPlatformTaskEmbedding_00022000.pth}
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export HEADLESS=${HEADLESS:-1}
export SEED=${SEED:-42}
export EPISODE_LENGTH=${EPISODE_LENGTH:-300}
export DEMO_AGENTS=6
export DEMO_CA_ENTMAX=0
unset TOKENHSI_INFERENCE_CONFIG
case ${SOLO_MASK:-1} in
    1) export DEMO_BLOCK_SOLO_FOR_JOINT=1; VARIANT=masked ;;
    0) export DEMO_BLOCK_SOLO_FOR_JOINT=0; VARIANT=baseline ;;
    *) echo "SOLO_MASK must be 0 or 1" >&2; exit 1 ;;
esac
export OUTPUT_PATH=${OUTPUT_PATH:-output/locoplatform_ca_solo_mask/$VARIANT}
exec bash "$SCRIPT_DIR/joint_carry_mixed_scene_demo_test.sh" "$CKPT"
