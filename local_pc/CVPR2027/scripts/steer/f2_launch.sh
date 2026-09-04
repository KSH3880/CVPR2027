#!/bin/bash
# Launch the F2 batch: 8 slots, one per GPU.
#
# One job per GPU is the default. Two fit and raise aggregate throughput about
# 7 % (measured: 8 x 35,214 solo vs 16 x 18,833 shared), but each slot then runs
# at half speed, so a batch takes 5.8 h instead of 3.1 h. That only pays when
# there are more than eight conditions worth running AT THE SAME TIME -- and
# usually there are not, because the input axes should be measured on top of
# whichever reward wins, not guessed alongside it. Go to two per GPU by pinning
# two launches to the same index, when a batch genuinely needs sixteen slots.
#
# The queue runner is bypassed on purpose: it treats a GPU with a live process
# as unavailable. Logs still go to runs/queue/logs so the analyzers find them.
set -u
cd /home/hwanhee/CVPR2027
LOGS=runs/queue/logs
mkdir -p "$LOGS" runs/steer

launch() {   # gpu tag cfg env-overrides...
    local gpu=$1 tag=$2 cfg=$3; shift 3
    CUDA_VISIBLE_DEVICES=$gpu STEER_TAG=$tag STEER_CFG=$cfg "$@" \
        nohup bash scripts/steer/f2_train.sh > "$LOGS/$tag.log" 2>&1 &
    printf "  gpu%s  %-18s %s\n" "$gpu" "$tag" "$*"
}

D="STEER_REAIM=1 STEER_LOOKAHEAD=1.2 STEER_XTRACK=1.0"   # the reference reward

echo "reward axis (input fixed at K=6, spacing 0.4; slot 1 is the control)"
launch 0 f2_1_ctrl      f2_base  env STEER_ZERO=1
launch 1 f2_3_D         f2_base  env $D
launch 2 f2_5_C         f2_base  env STEER_XTRACK=1.0
launch 3 f2_2_B         f2_base  env STEER_REAIM=1 STEER_LOOKAHEAD=1.2
launch 4 f2_4_D_la2.0   f2_base  env STEER_REAIM=1 STEER_LOOKAHEAD=2.0 STEER_XTRACK=1.0
launch 5 f2_6_D_noappr  f2_base  env $D STEER_APPROACH=0

echo "learning rate at the 4096 scale, and the seed noise floor"
launch 6 f2_14_D_lr707  f2_lr707 env $D
launch 7 f2_16_D_seed1  f2_base  env $D STEER_SEED=1

echo
echo "8 slots launched. progress: bash scripts/steer/progress.sh f2_ 3000"
