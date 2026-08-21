#!/bin/bash
# Measures FPS and peak GPU memory for each training knob, 8 GPUs, 12 iterations.
REPO=${TOKENHSI_REPO:-TokenHSI}
[ -d /home/hwanhee/CVPR2027/$REPO/tokenhsi ] || { echo "no such repo: $REPO"; exit 1; }

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd /home/hwanhee/CVPR2027/$REPO

CFGDIR=/home/hwanhee/CVPR2027/runs/sweep_cfgs/$REPO
LOGDIR=/home/hwanhee/CVPR2027/runs/sweep_logs/$REPO
mkdir -p $CFGDIR $LOGDIR
BASE=tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml

sed 's/mixed_precision: False/mixed_precision: True/' $BASE > $CFGDIR/amp.yaml
sed 's/minibatch_size: 16384/minibatch_size: 65536/' $BASE > $CFGDIR/mb64k.yaml
sed -e 's/mixed_precision: False/mixed_precision: True/' -e 's/minibatch_size: 16384/minibatch_size: 65536/' $BASE > $CFGDIR/all.yaml

run() {
    name=$1; cfg=$2; envs=$3
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > $LOGDIR/$name.mem 2>/dev/null &
    MEMPID=$!
    timeout 1200 bash /home/hwanhee/CVPR2027/scripts/train_multigpu.sh 8 tokenhsi/scripts/single_task/traj_train.sh \
        --cfg_train $cfg --num_envs $envs --max_iterations 12 > $LOGDIR/$name.log 2>&1
    kill $MEMPID 2>/dev/null
    fps=$(grep 'fps total' $LOGDIR/$name.log | tail -3 | awk '{s+=$NF;n++} END {printf "%.0f", s/n}')
    mem=$(sort -n $LOGDIR/$name.mem | tail -1)
    echo "$name | fps=$fps | peak_gpu_mem=${mem}MiB"
}

run base   $BASE            4096
run amp    $CFGDIR/amp.yaml 4096
run mb64k  $CFGDIR/mb64k.yaml 4096
run envs8k $BASE            8192
run all    $CFGDIR/all.yaml 8192
run same512 $CFGDIR/same_recipe.yaml 512
