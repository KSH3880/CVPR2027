#!/bin/bash
# Train/resume one slowdown-window experiment to step 3000, then evaluate every 200 PTH.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: c2_slowwin_to3000.sh <tag> <width-max> <init-or-dash> <detour-coef> <explicit-0|1>}
WIDTH_MAX=${2:?missing width-max}
INIT=${3:?missing init-or-dash}
DETOUR_COEF=${4:?missing detour-coef}
EXPLICIT=${5:?missing explicit flag}
EXECUTOR=${6:-ms18_maskteam_origscale_c06_s0}
TARGET=3000
TARGET_PTH="$ROOT/runs/coord/$TAG/coord_c2_003000.pth"
LOG="$ROOT/runs/queue/logs/${TAG}_to3000.log"

if [ "$INIT" = - ]; then
    START=0
    RESUME_ENV=()
else
    INIT=$(realpath "$INIT")
    base=$(basename "$INIT")
    if ! [[ "$base" =~ coord_c2_([0-9]{6})[.]pth$ ]]; then
        echo "resume PTH 이름에서 step을 읽을 수 없다: $INIT" >&2
        exit 2
    fi
    START=$((10#${BASH_REMATCH[1]}))
    RESUME_ENV=(COORD_INIT="$INIT" COORD_RESUME_IN_PLACE=1)
fi
if [ "$START" -ge "$TARGET" ] && [ ! -f "$TARGET_PTH" ]; then
    echo "resume step은 target보다 작아야 한다: $START" >&2
    exit 2
fi

if [ ! -f "$TARGET_PTH" ]; then
    REMAIN=$((TARGET - START))
    if [ "$EXPLICIT" = 1 ]; then
        EXPLICIT_ENV=(
            COORD_C2_EXPLICIT_GAP_COEF=10
            COORD_C2_EXPLICIT_ANCHOR_COEF=3
            COORD_C2_EXPLICIT_POST_COEF=3
            COORD_C2_INITIAL_PLAN_WEIGHT=3
        )
    else
        EXPLICIT_ENV=(
            COORD_C2_EXPLICIT_GAP_COEF=0
            COORD_C2_EXPLICIT_ANCHOR_COEF=0
            COORD_C2_EXPLICIT_POST_COEF=0
            COORD_C2_INITIAL_PLAN_WEIGHT=1
        )
    fi
    echo "C2_TO3000_TRAIN tag=$TAG start=$START target=$TARGET detour=$DETOUR_COEF explicit=$EXPLICIT" >> "$LOG"
    env "${RESUME_ENV[@]}" "${EXPLICIT_ENV[@]}" \
        COORD_C2_DETOUR_DELAY_COEF="$DETOUR_COEF" \
        COORD_ITERS="$REMAIN" COORD_SAVE_EVERY=200 COORD_ENVS=64 \
        COORD_SKIP_FINAL_EVAL=1 \
        bash "$ROOT/scripts/coord/c2_slowwin_compare.sh" \
            "$TAG" "$WIDTH_MAX" "$EXECUTOR" >> "$LOG" 2>&1
fi

for step in $(seq 200 200 "$TARGET"); do
    printf -v padded '%06d' "$step"
    checkpoint="$ROOT/runs/coord/$TAG/coord_c2_${padded}.pth"
    if [ ! -f "$checkpoint" ]; then
        echo "C2_TO3000_MISSING checkpoint=$checkpoint" >> "$LOG"
        exit 3
    fi
    echo "C2_TO3000_EVAL step=$step checkpoint=$checkpoint" >> "$LOG"
    COORD_SOURCE="$TAG" bash "$ROOT/scripts/coord/eval_one.sh" \
        "$checkpoint" cross 128 0 >> "$LOG" 2>&1
done
echo "C2_TO3000_DONE tag=$TAG" >> "$LOG"
