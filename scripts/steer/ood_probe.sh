#!/bin/bash
# Evaluate a trained policy on path distributions it never trained on.
#
#   bash scripts/steer/ood_probe.sh <gpu> <out_tag> <ckpt> <src_env_tag> KEY=VAL ...
#
# Why: every number we have is scored on the SAME v3 generator the policy trained
# on, so "follows the commanded path" and "follows v3-shaped paths" are not yet
# distinguished. The six v3 shape constants were hand-picked, never measured, and
# the co-op avoidance paths this feeds into will not share their statistics.
# Training is not needed -- only the eval-time generator changes.
set -u
ROOT=/home/hwanhee/CVPR2027; cd $ROOT
GPU=$1; OUT=$2; CKPT=$3; SRC=$4; shift 4
set -a; . "$ROOT/runs/queue/logs/$SRC.env" 2>/dev/null || true; set +a
for kv in "$@"; do export "$kv"; done
CUDA_VISIBLE_DEVICES=$GPU STEER_CKPT=$CKPT STEER_TAG=$OUT STEER_CELL=60 STEER_LOG=900 \
    bash scripts/steer/f5_eval.sh 512 > "runs/queue/logs/eval_$OUT.log" 2>&1
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh; conda activate tokenhsi
python scripts/steer/record_result.py "$OUT" >> runs/queue/results.tsv
