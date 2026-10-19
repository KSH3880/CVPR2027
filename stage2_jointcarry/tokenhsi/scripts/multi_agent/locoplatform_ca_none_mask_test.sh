#!/bin/bash
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
if [ $# -gt 1 ]; then
    echo "usage: $0 [LocoPlatform-checkpoint.pth]" >&2
    exit 1
fi
CKPT=${1:-output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1LocoPlatformTaskEmbedding_13-16-32-42/nn/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1LocoPlatformTaskEmbedding_00012500.pth}
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export HEADLESS=${HEADLESS:-1}
export SEED=${SEED:-42}
export DEMO_CA_ENTMAX=0
SCENE=${SCENE:-mixed}
case ${NONE_MASK:-1} in
    1) export TOKENHSI_INFERENCE_CONFIG="$ROOT/tokenhsi/data/cfg/inference/locoplatform_ca_none_mask.yaml"; VARIANT=masked ;;
    0) unset TOKENHSI_INFERENCE_CONFIG; VARIANT=baseline ;;
    *) echo "NONE_MASK must be 0 or 1" >&2; exit 1 ;;
esac
export OUTPUT_PATH=${OUTPUT_PATH:-output/locoplatform_ca_none_mask/$SCENE/$VARIANT}
case "$SCENE" in
    mixed)
        exec bash "$SCRIPT_DIR/joint_carry_mixed_scene_demo_test.sh" "$CKPT" ;;
    two_agents)
        export TASK_GRAPH=${TASK_GRAPH:-random_scenario}
        export EVAL_SKILLS=${EVAL_SKILLS:-carryWith}
        exec bash "$SCRIPT_DIR/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_test.sh" "$CKPT" 2 "${NUM_ENVS:-16}" 4 "${REPEATS:-3}" ;;
    *) echo "SCENE must be mixed or two_agents" >&2; exit 1 ;;
esac
