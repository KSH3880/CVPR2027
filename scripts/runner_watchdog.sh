#!/bin/bash
# Keeps runner.py alive. Restarts it if it dies; logs every restart.
ROOT=/home/hwanhee/CVPR2027
LOG=$ROOT/runs/queue/watchdog.log
mkdir -p "$(dirname "$LOG")"

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

echo "[$(date '+%F %T')] watchdog up (pid $$)" >> "$LOG"
while true; do
    if ! pgrep -u "$USER" -f "python .*scripts/runner\.py" > /dev/null; then
        echo "[$(date '+%F %T')] runner not running - starting" >> "$LOG"
        nohup python "$ROOT/scripts/runner.py" >> "$ROOT/runs/queue/runner.log" 2>&1 &
        sleep 5
    fi
    sleep 30
done
