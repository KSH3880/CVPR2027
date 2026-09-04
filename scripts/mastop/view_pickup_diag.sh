#!/bin/bash
# Compare pickup retention while isolating the B layout from B fine-tuning.
set -euo pipefail

ROOT=/home/hwanhee/koo_cvpr
MODE=${1:-}

A_CKPT=$ROOT/runs/output/masteer_pickup/pickupdiag_live_15k_s0/Humanoid_26-14-37-49/nn/Humanoid_00006000.pth
B_CKPT=$ROOT/TokenHSI-mastop/output/mastop_carry/mastop_ab_gradual_live6k_s0_r1/Humanoid_27-14-42-59/nn/Humanoid.pth

case "$MODE" in
    1)
        LABEL="A checkpoint / B layout / stop disabled"
        CKPT=$A_CKPT
        export MSTOP_MIX=0,1,0,0
        export MSTOP_SOLO_AFTER=1000000
        export MA_SEP=0
        ;;
    2)
        LABEL="B checkpoint / original A layout"
        CKPT=$B_CKPT
        export MSTOP_MIX=1,0,0,0
        unset MSTOP_SOLO_AFTER
        export MA_SEP=9
        ;;
    *)
        echo "usage: bash scripts/mastop/view_pickup_diag.sh <1|2>" >&2
        echo "  1: A checkpoint / B layout / stop disabled" >&2
        echo "  2: B checkpoint / original A layout" >&2
        exit 2
        ;;
esac

test -f "$CKPT"

export TOKENHSI_REPO=TokenHSI-mastop
export MSTOP_STAGE=B MSTOP_SEED=0 MS_SEED=0
export MS_MRAND=0 MS_VEL_W=1 MS_CLIP=0
export MS_VIZ=1 MS_CAM=top MS_CAM_H=17 MS_CAM_B=9
export MA_SPAWN_GAP=1.0 MA_C=0 MA_BETA=0
export MA_TOKEN=live MA_TOKENIZER_ZERO=1 MSTOP_BRAKE_T=0.8

echo "pickup diagnostic: $LABEL"
echo "checkpoint: $CKPT"
echo "GPU: physical 6, port: 6100"

cd "$ROOT"
exec bash scripts/tokenhsi_gui.sh --port 6100 --gpu 6 \
    /home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python -u ./tokenhsi/run.py \
    --task HumanoidMAStopCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env output/mastop_carry/configs/mastop_ab_gradual_live6k_s0_r1.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth" \
    --checkpoint "$CKPT" \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
    --test --eval --eval_task carry --num_envs 1 --seed 0
