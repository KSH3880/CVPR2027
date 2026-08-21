#!/bin/bash
# Verifies train.sh actually runs on 1/2/4/8 GPUs with the recipe-preserving mode.
LOG=/home/hwanhee/CVPR2027/runs/compat_logs
mkdir -p $LOG
for N in 1 2 4 8; do
    timeout 1500 bash /home/hwanhee/CVPR2027/scripts/train.sh $N \
        tokenhsi/scripts/single_task/traj_train.sh --max_iterations 10 \
        > $LOG/gpu$N.log 2>&1
    fps=$(grep 'fps total' $LOG/gpu$N.log | tail -3 | awk '{s+=$NF;n++} END {if(n>0) printf "%.0f", s/n; else print "FAIL"}')
    iters=$(grep -c 'fps total' $LOG/gpu$N.log)
    echo "$N GPU | fps=$fps | iterations_logged=$iters"
done
