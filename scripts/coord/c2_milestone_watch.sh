#!/bin/bash
# Evaluate deterministic Cross behavior while C2 training keeps running.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?사용법: c2_milestone_watch.sh <tag> [step ...]}
shift
if [ "$#" -eq 0 ]; then set -- 50 100 250; fi
LOG="$ROOT/runs/queue/logs/${TAG}_milestone_eval.log"
GPU_LOCK="$ROOT/runs/queue/coord_eval_gpu.lock"

coord_gpu_jobs() {
    ps -eo comm=,args= | awk '
        $1 ~ /^python/ &&
        ($0 ~ /coordinator\.train_closed_loop/ ||
         $0 ~ /tokenhsi\/run\.py.*HumanoidMACoordCarry/) { count += 1 }
        END { print count + 0 }
    '
}

run_eval_with_slot() {
    local checkpoint=$1
    (
        flock 9
        while [ "$(coord_gpu_jobs)" -ge 4 ]; do
            echo "C2_MILESTONE_GPU_WAIT active=$(coord_gpu_jobs) checkpoint=$checkpoint" >> "$LOG"
            sleep 20
        done
        COORD_SOURCE="$TAG" bash "$ROOT/scripts/coord/eval_one.sh" \
            "$checkpoint" cross 128 0 >> "$LOG" 2>&1
    ) 9>"$GPU_LOCK"
}

for step in "$@"; do
    printf -v padded '%06d' "$step"
    checkpoint="$ROOT/runs/coord/$TAG/coord_c2_${padded}.pth"
    while [ ! -f "$checkpoint" ]; do
        if ! pgrep -f "coordinator.train_closed_loop.*runs/coord/$TAG" >/dev/null; then
            echo "C2_MILESTONE_STOP missing=$checkpoint" >> "$LOG"
            exit 3
        fi
        sleep 20
    done
    echo "C2_MILESTONE_EVAL step=$step checkpoint=$checkpoint" >> "$LOG"
    run_eval_with_slot "$checkpoint"
done
