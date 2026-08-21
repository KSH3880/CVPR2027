#!/bin/bash
REPO=${TOKENHSI_REPO:-TokenHSI}
[ -d /home/hwanhee/CVPR2027/$REPO/tokenhsi ] || { echo "no such repo: $REPO"; exit 1; }

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd /home/hwanhee/CVPR2027/$REPO
LOGDIR=/home/hwanhee/CVPR2027/runs/verify_logs/$REPO
mkdir -p $LOGDIR
for name in carry_test traj_test stage2_longterm_test stage2_object_chair_test stage2_object_table_test stage2_terrain_carry_test stage2_terrain_traj_test; do
    s=$(ls tokenhsi/scripts/single_task/$name.sh tokenhsi/scripts/tokenhsi/$name.sh 2>/dev/null | head -1)
    cmd="$(grep -v '^#!' $s) --headless"
    timeout 900 bash -c "$cmd" > $LOGDIR/$name.log 2>&1
    if grep -q "^reward:" $LOGDIR/$name.log; then
        echo "PASS  $name   $(grep '^reward:' $LOGDIR/$name.log | tail -1)"
    else
        echo "FAIL  $name   $(grep -m1 'RuntimeError\|Traceback' $LOGDIR/$name.log | head -c 120)"
    fi
done
