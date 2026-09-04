#!/bin/bash
# Runs every *_test.sh from the README headless for a short time and reports pass/fail.
#
# Defaults to the TokenHSI baseline. Pick another repo with TOKENHSI_REPO:
#   TOKENHSI_REPO=TokenHSI-steer bash verify_all.sh
REPO=${TOKENHSI_REPO:-TokenHSI}
[ -d /home/hwanhee/CVPR2027/$REPO/tokenhsi ] || { echo "no such repo: $REPO"; exit 1; }

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd /home/hwanhee/CVPR2027/$REPO

LOGDIR=/home/hwanhee/CVPR2027/runs/verify_logs/$REPO
mkdir -p $LOGDIR

for s in tokenhsi/scripts/single_task/*_test.sh tokenhsi/scripts/tokenhsi/*_test.sh; do
    name=$(basename $s .sh)
    cmd="$(grep -v '^#!' $s) --headless"
    timeout 200 bash -c "$cmd" > $LOGDIR/$name.log 2>&1
    if grep -q "^reward:" $LOGDIR/$name.log; then
        echo "PASS  $name   $(grep -m1 '^reward:' $LOGDIR/$name.log)"
    else
        echo "FAIL  $name   $(grep -m1 'Error\|error:' $LOGDIR/$name.log | head -c 150)"
    fi
done
