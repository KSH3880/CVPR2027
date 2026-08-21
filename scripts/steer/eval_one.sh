#!/bin/bash
# The only way to start an evaluation. Takes the claim first, so a second
# launcher cannot start the same one.
#
#   bash scripts/steer/eval_one.sh <tag> <gpu> [envs]
#
# Two evaluations of the same tag write the same log and the same npz, and the
# result is a run with two different success rates and a scrambled trajectory
# dump. That happened twice -- once because autofill never wrote the claim it
# checks for, once because a hand-launched eval bypassed the claim entirely.
# mkdir is atomic, so taking the claim here closes both holes at once: autofill
# and a human at a terminal now go through the same door.
set -u
ROOT=/home/hwanhee/CVPR2027
TAG=${1:?usage: eval_one.sh <tag> <gpu> [envs] [repeat_seed]}
GPU=${2:?usage: eval_one.sh <tag> <gpu> [envs] [repeat_seed]}
ENVS=${3:-512}
# A repeat re-scores the SAME checkpoint on a different draw of paths, and
# writes under its own tag. Rankings across F4 all rest on one measurement each,
# and this is the cheap half of finding out how much of that ranking is real:
# 5 minutes per repeat instead of a 3.3 h retrain.
RSEED=${4:-}
OUT=${RSEED:+${TAG}_r$RSEED}; OUT=${OUT:-$TAG}
CLAIM=$ROOT/runs/queue/gpu_locks/eval_$OUT
SOURCE=${QUEUE_EVAL_SOURCE:-$TAG}

mkdir -p "$ROOT/runs/queue/gpu_locks"
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "eval_one: $TAG is already claimed by $(cat "$CLAIM/gpu" 2>/dev/null || echo '?')" >&2
    exit 3
fi
echo "$GPU" > "$CLAIM/gpu"
trap 'rm -rf "$CLAIM"' EXIT

CK=${QUEUE_EVAL_CKPT:-}
[ -n "$CK" ] || CK=$(ls -d "$ROOT"/TokenHSI-steer/output/steer/"$SOURCE"/*/nn/Humanoid.pth 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "eval_one: no checkpoint for $TAG" >&2; exit 4; }

# Replay what the run trained with. Without this the env is built from today's
# defaults: STEER_K=3 then crashes on a shape mismatch, and a STEER_TRAJ default
# changed after training silently scores the policy on paths it never saw.
ENVFILE=$ROOT/runs/queue/logs/$SOURCE.env
if [ -f "$ENVFILE" ]; then
    set -a; . "$ENVFILE"; set +a
else
    echo "eval_one: no $SOURCE.env -- refusing to guess the training settings" >&2
    exit 5
fi

[ -n "$RSEED" ] && export STEER_SEED=$RSEED   # after the sidecar, so it wins

# 학습이 자기 env cfg 를 생성했으면(envSpacing·접촉쌍을 바꾼 런) 평가도 그걸 쓴다.
# 사이드카(.env)는 STEER_* 환경변수만 되살리므로 cfg 파일 자체는 이 줄이 없으면
# 오늘의 기본값으로 지어진다 -- F8~F12 를 통째로 날린 그 불일치다.
GENCFG=$ROOT/runs/gen_cfgs/steer/env_$SOURCE.yaml
[ -f "$GENCFG" ] && export STEER_ENVCFG=$GENCFG

cd "$ROOT"
CUDA_VISIBLE_DEVICES=$GPU STEER_CKPT=$CK STEER_TAG=$OUT STEER_CELL=60 STEER_LOG=900 \
    bash scripts/steer/f5_eval.sh "$ENVS" > "runs/queue/logs/eval_$OUT.log" 2>&1

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
python scripts/steer/record_result.py "$OUT" >> runs/queue/results.tsv
