#!/bin/bash
# One coarse C1 comparison: finish legacy baseline, evaluate it, then train and
# evaluate the continuous-constraint loss.  Model architecture stays fixed.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_TAG=${1:-c1_cross_s0}
NEXT_TAG=${2:-c1_contcost_s0}
EXECUTOR=${3:-ms18_maskteam_origscale_c06_s0}
BASE_PID_FILE="$ROOT/runs/queue/logs/$BASE_TAG.pid"
BASE_FINAL="$ROOT/runs/coord/$BASE_TAG/coord_c1_000500.pth"

if [ ! -f "$BASE_PID_FILE" ]; then echo "기준 PID 없음: $BASE_PID_FILE" >&2; exit 2; fi
base_pid=$(tr -d '[:space:]' < "$BASE_PID_FILE")
while [ ! -f "$BASE_FINAL" ]; do
    if ! kill -0 "$base_pid" 2>/dev/null; then
        echo "기준 학습이 500 PTH 없이 종료됨: $BASE_TAG" >&2
        exit 3
    fi
    sleep 30
done
while kill -0 "$base_pid" 2>/dev/null; do sleep 2; done

echo "[overnight] baseline complete: $BASE_FINAL"
bash "$ROOT/scripts/coord/eval_one.sh" "$BASE_TAG" cross 128 0 || echo "[overnight] baseline cross eval failed"
bash "$ROOT/scripts/coord/eval_one.sh" "$BASE_TAG" free 128 0 || echo "[overnight] baseline free eval failed"

if [ -e "$ROOT/runs/coord/$NEXT_TAG" ]; then
    echo "다음 태그가 이미 존재함: $NEXT_TAG" >&2
    exit 4
fi
echo "[overnight] starting continuous-constraint comparison: $NEXT_TAG"
MS_SCEN=cross COORD_ITERS=500 COORD_SAVE_EVERY=10 COORD_ENVS=64 \
    bash "$ROOT/scripts/coord/train_local.sh" "$NEXT_TAG" "$EXECUTOR"

bash "$ROOT/scripts/coord/eval_one.sh" "$NEXT_TAG" cross 128 0 || echo "[overnight] next cross eval failed"
bash "$ROOT/scripts/coord/eval_one.sh" "$NEXT_TAG" free 128 0 || echo "[overnight] next free eval failed"
echo "[overnight] comparison complete"
