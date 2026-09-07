#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$repo_root/TokenHSI-masteer"
run_dir="$repo_root/runs/trajectory_predictor"
python_bin="${TP_PYTHON:-/home/cvlab/anaconda3/envs/tokenhsi/bin/python}"
epochs="${TP_EPOCHS:-100}"

mkdir -p "$run_dir/checkpoints" "$run_dir/videos" "$run_dir/logs"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=""
exec "$python_bin" -m trajectory_predictor.watch_videos \
  --checkpoint-dir "$run_dir/checkpoints" \
  --data "$run_dir/data/joint_v1_val.pt" \
  --output-dir "$run_dir/videos" \
  --final-epoch "$epochs" \
  --poll-seconds 5 \
  --scenes 4 \
  --fps 10 \
  --max-frames 120
