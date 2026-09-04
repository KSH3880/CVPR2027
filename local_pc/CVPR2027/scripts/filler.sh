#!/bin/bash
# Runs cheap no-training jobs alongside the main queue.
#
# The eight training slots hold their GPUs for hours but leave ~80 GB and most
# of the SM time free, so short probes ride along without displacing them.
# One at a time, rotating GPUs, so the added load stays small.
ROOT=/home/hwanhee/CVPR2027
BACK=$ROOT/runs/queue/backlog
DONE=$ROOT/runs/queue/backlog_done
LOG=$ROOT/runs/queue/logs
mkdir -p "$BACK" "$DONE" "$LOG"

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd "$ROOT"

gpu=0
while true; do
    job=$(ls "$BACK"/*.json 2>/dev/null | head -1)
    if [ -z "$job" ]; then sleep 120; continue; fi
    name=$(python3 -c "import json;print(json.load(open('$job'))['name'])")
    cmd=$(python3 -c "import json;print(json.load(open('$job'))['cmd'])")
    envs=$(python3 -c "
import json; j=json.load(open('$job'))
print(' '.join(f'{k}={v}' for k,v in j.get('env',{}).items()))")
    echo "[$(date '+%T')] filler start $name on gpu $gpu" >> "$ROOT/runs/queue/filler.log"
    ( CUDA_VISIBLE_DEVICES=$gpu env $envs timeout 2400 bash -c "$cmd" > "$LOG/$name.log" 2>&1 )
    code=$?
    echo "[$(date '+%T')] filler done $name exit=$code" >> "$ROOT/runs/queue/filler.log"
    mv "$job" "$DONE/" 2>/dev/null
    gpu=$(( (gpu + 1) % 8 ))
done
