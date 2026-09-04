#!/bin/bash
# Finishes the interrupted benchmark: compat 4/8 GPU + the mixed_precision retry.
OUT=/home/hwanhee/CVPR2027/runs/results/finish_results.txt
LOG=/home/hwanhee/CVPR2027/runs/compat_logs
CFG=/home/hwanhee/CVPR2027/runs/sweep_cfgs/TokenHSI
mkdir -p $LOG $CFG
: > $OUT

for N in 4 8; do
    timeout 1500 bash /home/hwanhee/CVPR2027/scripts/train.sh $N \
        tokenhsi/scripts/single_task/traj_train.sh --max_iterations 10 > $LOG/gpu$N.log 2>&1
    fps=$(grep 'fps total' $LOG/gpu$N.log | tail -3 | awk '{s+=$NF;n++} END {if(n>0) printf "%.0f", s/n; else print "FAIL"}')
    echo "compat ${N}GPU(same) | fps=$fps | iters=$(grep -c 'fps total' $LOG/gpu$N.log)" >> $OUT
done

sed 's/mixed_precision: False/mixed_precision: True/' \
    /home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml > $CFG/amp.yaml
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > $LOG/amp.mem 2>/dev/null &
MEMPID=$!
timeout 1500 bash /home/hwanhee/CVPR2027/scripts/train_multigpu.sh 8 \
    tokenhsi/scripts/single_task/traj_train.sh --cfg_train $CFG/amp.yaml --max_iterations 12 \
    > $LOG/amp8.log 2>&1
kill $MEMPID 2>/dev/null
fps=$(grep 'fps total' $LOG/amp8.log | tail -3 | awk '{s+=$NF;n++} END {if(n>0) printf "%.0f", s/n; else print "FAIL"}')
echo "amp 8GPU(fast) | fps=$fps | peak_gpu_mem=$(sort -n $LOG/amp.mem | tail -1)MiB" >> $OUT
cat $OUT
