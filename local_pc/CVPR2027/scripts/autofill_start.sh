#!/bin/bash
# Start the autofill loop detached, and only if one is not already running.
#
# Starting it inline from an interactive shell keeps losing it: the loop ends up
# a child of that shell, so it dies with the shell, and `kill $(pgrep -f ...)`
# has twice matched the calling shell itself. setsid + a running-check avoids
# both, and makes this safe to call repeatedly (e.g. from a watchdog).
set -u
cd /home/hwanhee/CVPR2027
if pgrep -f "[a]utofill.py --loop" > /dev/null; then
    echo "autofill already running (pid $(pgrep -f '[a]utofill.py --loop' | head -1))"
    exit 0
fi
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
setsid nohup python scripts/autofill.py --loop --interval 60 \
    >> runs/queue/autofill.log 2>&1 < /dev/null &
disown
sleep 3
pid=$(pgrep -f "[a]utofill.py --loop" | head -1)
echo "autofill started (pid ${pid:-FAILED})"
