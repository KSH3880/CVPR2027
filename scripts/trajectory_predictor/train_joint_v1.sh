#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$repo_root/TokenHSI-masteer"
run_dir="$repo_root/runs/trajectory_predictor"
python_bin="${TP_PYTHON:-/home/cvlab/anaconda3/envs/tokenhsi/bin/python}"
gpu="${TP_GPU:-0}"
epochs="${TP_EPOCHS:-100}"
save_every="${TP_SAVE_EVERY:-5}"
batch_size="${TP_BATCH_SIZE:-256}"
workers="${TP_WORKERS:-0}"

mkdir -p "$run_dir/data" "$run_dir/checkpoints" "$run_dir/videos" "$run_dir/logs"
exec 9>"$run_dir/joint_v1_train.lock"
if ! flock -n 9; then
  echo "joint_v1 generation/training is already running" >&2
  exit 4
fi

cd "$project_dir"
for split in train val test; do
  data_file="$run_dir/data/joint_v1_${split}.pt"
  if [[ -s "$data_file" ]]; then
    echo "using existing dataset: $data_file"
  else
    "$python_bin" -m trajectory_predictor.generate_dataset \
      --split "$split" \
      --output "$data_file"
  fi
done

export CUDA_VISIBLE_DEVICES="$gpu"
exec "$python_bin" -m trajectory_predictor.train \
  --train "$run_dir/data/joint_v1_train.pt" \
  --val "$run_dir/data/joint_v1_val.pt" \
  --output-dir "$run_dir/checkpoints" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --workers "$workers" \
  --save-every "$save_every" \
  --device cuda
