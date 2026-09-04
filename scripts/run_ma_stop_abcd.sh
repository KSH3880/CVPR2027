#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=$ROOT/TokenHSI-mastop
TRAIN=$REPO/tokenhsi/scripts/tokenhsi/stage2_ma_stop_carry_train.sh
EVAL=$REPO/tokenhsi/scripts/tokenhsi/stage2_ma_stop_carry_eval.sh
BASE=$ROOT/runs/output/masteer_pickup/pickupdiag_live_15k_s0/Humanoid_26-14-37-49/nn/Humanoid_00006000.pth
ITERS_B=${MSTOP_ITERS_B:-3000}
ITERS_C=${MSTOP_ITERS_C:-3000}
ITERS_D=${MSTOP_ITERS_D:-3000}
RUN_ID=${MSTOP_RUN_ID:-gradual_live6k_s0_r1}
B_TAG=mastop_ab_${RUN_ID}
C_TAG=mastop_abc_${RUN_ID}
D_TAG=mastop_abcd_${RUN_ID}

latest() {
    find "$REPO/output/mastop_carry/$1" -name Humanoid.pth -printf "%T@ %p\n" |
        sort -rn | head -1 | cut -d" " -f2-
}


echo "MSTOP_CHAIN A+B"
MSTOP_STAGE=B MSTOP_TAG="$B_TAG" MSTOP_START_CKPT="$BASE"     MSTOP_ITERS="$ITERS_B" "$TRAIN"
B_CKPT=$(latest "$B_TAG")
MSTOP_STAGE=B MSTOP_TAG="$B_TAG" MSTOP_GPU=6 "$EVAL" "$B_CKPT"

echo "MSTOP_CHAIN A+B+C"
MSTOP_STAGE=C MSTOP_TAG="$C_TAG" MSTOP_START_CKPT="$B_CKPT"     MSTOP_ITERS="$ITERS_C" "$TRAIN"
C_CKPT=$(latest "$C_TAG")
MSTOP_STAGE=C MSTOP_TAG="$C_TAG" MSTOP_GPU=6 "$EVAL" "$C_CKPT"

echo "MSTOP_CHAIN A+B+C+D"
MSTOP_STAGE=D MSTOP_TAG="$D_TAG" MSTOP_START_CKPT="$C_CKPT"     MSTOP_ITERS="$ITERS_D" "$TRAIN"
D_CKPT=$(latest "$D_TAG")
MSTOP_STAGE=D MSTOP_TAG="$D_TAG" MSTOP_GPU=6 "$EVAL" "$D_CKPT"
echo "MSTOP_CHAIN_DONE checkpoint=$D_CKPT"
