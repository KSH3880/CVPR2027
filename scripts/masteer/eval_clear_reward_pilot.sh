#!/bin/bash
# Evaluate the initial policy and the first five ~100-iteration pilot archives.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: eval_clear_reward_pilot.sh <tag> <gpu:6|7> [envs]}
GPU=${2:?usage: eval_clear_reward_pilot.sh <tag> <gpu:6|7> [envs]}
ENVS=${3:-512}
case "$GPU" in 6|7) ;; *) echo "GPU must be 6 or 7" >&2; exit 2 ;; esac

ENV_FILE="$ROOT/runs/queue/logs/$TAG.env"
OUT="$ROOT/TokenHSI-masteer/output/masteer/$TAG"
if [ ! -f "$ENV_FILE" ]; then
    echo "pilot sidecar 없음: $ENV_FILE" >&2
    exit 4
fi

BASE_SUFFIX=pilot_initial
BASE_METRICS="$ROOT/runs/results/masteer/eval_${TAG}__${BASE_SUFFIX}.npy"
if [ "${PILOT_REFRESH_BASELINE:-1}" = 1 ] || [ ! -f "$BASE_METRICS" ]; then
    STACK_STAGE_SUFFIX="$BASE_SUFFIX" \
        bash "$ROOT/scripts/masteer/stack_stage_eval.sh" "$TAG" "$GPU" initial "$ENVS"
fi

mapfile -t EPOCHS < <(
    find "$OUT" -type f -name 'Humanoid_[0-9]*.pth' -printf '%f\n' 2>/dev/null |
        sed -n 's/^Humanoid_0*\([0-9][0-9]*\)\.pth$/\1/p' |
        sort -n | head -n 5
)
if [ "${#EPOCHS[@]}" -lt 5 ]; then
    echo "100~500 pilot archive가 5개보다 적음: ${#EPOCHS[@]}개" >&2
fi

nominal=100
for epoch in "${EPOCHS[@]}"; do
    suffix="pilot_${nominal}"
    STACK_STAGE_SUFFIX="$suffix" \
        bash "$ROOT/scripts/masteer/stack_stage_eval.sh" "$TAG" "$GPU" "$epoch" "$ENVS"
    metrics="$ROOT/runs/results/masteer/eval_${TAG}__${suffix}.npy"
    printf 'PILOT_CHECK nominal=%d epoch=%d\n' "$nominal" "$epoch"
    python3 "$ROOT/scripts/masteer/clear_reward_pilot_guard.py" \
        "$metrics" "$BASE_METRICS"
    nominal=$((nominal + 100))
done
