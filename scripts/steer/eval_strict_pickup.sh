#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REFERENCE_ROOT=/home/hwanhee/CVPR2027
TAG=f22b_long_c20_e25000
GPU=7
ENVS=512
[ "$#" -lt 1 ] || TAG=$1
[ "$#" -lt 2 ] || GPU=$2
[ "$#" -lt 3 ] || ENVS=$3
SOURCE_TAG=f22b_long_c20
ITER=25000
OUT="$TAG""__strictpick"

SOURCE_REPO=$REFERENCE_ROOT/TokenHSI-steer
ENV_FILE=$REFERENCE_ROOT/runs/queue/logs/$SOURCE_TAG.env
SOURCE_ENV=$REFERENCE_ROOT/runs/gen_cfgs/steer/env_$SOURCE_TAG.yaml
SOURCE_CK=$SOURCE_REPO/output/steer/$SOURCE_TAG/Humanoid_19-22-50-01/nn/Humanoid_00025000.pth
BASE_CK=$SOURCE_REPO/output/tokenhsi/ckpt_stage1.pth

for path in "$SOURCE_REPO/tokenhsi/run.py" "$ENV_FILE" "$SOURCE_ENV" "$SOURCE_CK" "$BASE_CK"; do
    [ -e "$path" ] || { echo "missing reference input: $path" >&2; exit 4; }
done

CLAIM=$ROOT/runs/queue/gpu_locks/eval_$OUT
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "steer strict eval: $OUT is already claimed" >&2
    exit 3
fi
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT

LOCAL_CK_DIR=$ROOT/runs/checkpoints/steer/$SOURCE_TAG
LOCAL_CK=$LOCAL_CK_DIR/Humanoid_00025000.pth
LOCAL_ENV=$ROOT/runs/gen_cfgs/steer/env_$SOURCE_TAG.yaml
LOG=$ROOT/runs/queue/logs/eval_$OUT.log
NPZ=$ROOT/runs/steer/$OUT.npz
JSON=$ROOT/runs/results/steer/$OUT.json
mkdir -p "$LOCAL_CK_DIR" "$(dirname "$LOCAL_ENV")" "$(dirname "$LOG")" "$(dirname "$NPZ")" "$(dirname "$JSON")" "$ROOT/runs/output/steer_strict/$OUT"
[ -f "$LOCAL_CK" ] || cp --reflink=auto "$SOURCE_CK" "$LOCAL_CK"
cp "$SOURCE_ENV" "$LOCAL_ENV"

set -a
source "$ENV_FILE"
set +a
export STEER_LOG=2000
export STEER_OUT=$ROOT/runs/steer
export STEER_TAG=$OUT
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU

cd "$SOURCE_REPO"
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi_koo

python -u ./tokenhsi/run.py --task HumanoidSteerCarry --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml --cfg_env "$LOCAL_ENV" --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml --hrl_checkpoint "$BASE_CK" --checkpoint "$LOCAL_CK" --output_path "$ROOT/runs/output/steer_strict/$OUT" --test --eval --eval_task carry --num_envs "$ENVS" --headless > "$LOG" 2>&1

cd "$ROOT"
python scripts/steer/strict_pickup.py "$NPZ" --json "$JSON" | tee -a "$LOG"
