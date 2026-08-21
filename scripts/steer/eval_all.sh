#!/bin/bash
# Evaluate finished runs, one per free GPU, then score them together.
#   bash scripts/steer/eval_all.sh f2_1_ctrl f2_3_D ...
#
# Each eval needs the SAME STEER_* the policy trained under, so the settings are
# read back out of the training log rather than assumed -- an eval run under the
# wrong lookahead or K silently measures a different task.
set -u
cd /home/hwanhee/CVPR2027
LOGS=runs/queue/logs
GPUS=(${EVAL_GPUS:-0 1 2 6})
i=0

for tag in "$@"; do
    ck=$(find TokenHSI-steer/output/steer/"$tag" -name Humanoid.pth 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "skip $tag (no checkpoint)"; continue; }
    # recover the training-time knobs from the launcher line in the log dir name
    src=$(grep -o "STEER_[A-Z_]*=[^ ]*" "$LOGS/$tag.log" 2>/dev/null | sort -u | tr '\n' ' ')
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    i=$((i+1))
    echo "gpu$gpu  eval $tag"
    CUDA_VISIBLE_DEVICES=$gpu STEER_CKPT="$(realpath "$ck")" STEER_TAG="$tag" \
        STEER_CELL=60 STEER_LOG=900 \
        nohup bash scripts/steer/f2_eval.sh 512 > "$LOGS/eval_$tag.log" 2>&1 &
    # claim the GPU so autofill does not stack a training job on top during the
    # ~30 s of IsaacGym startup, when nvidia-smi still reports the card as free
    mkdir -p runs/queue/gpu_locks && echo $! > runs/queue/gpu_locks/gpu$gpu
done
wait
echo
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
python scripts/steer/analyze_full.py $(for t in "$@"; do echo runs/steer/$t.npz; done)
