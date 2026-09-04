#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
ROOT=$(cd "$REPO/.." && pwd)
cd "$REPO"

STAGE=${MSTOP_STAGE:?MSTOP_STAGE=B,C,D}
case "$STAGE" in B|C|D) ;; *) echo "train stage must be B, C, or D" >&2; exit 2;; esac
TAG=${MSTOP_TAG:-mastop_${STAGE,,}_s0}
START=${MSTOP_START_CKPT:?MSTOP_START_CKPT is required}
BASE=${MSTOP_BASE_CKPT:-$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth}
ITERS=${MSTOP_ITERS:-3000}
ENVS=${MSTOP_ENVS:-2048}
SEED=${MSTOP_SEED:-0}
MB=${MSTOP_MB:-8192}
AMP_MB=${MSTOP_AMP_MB:-2048}
if [ ! -f "$START" ] || [ ! -f "$BASE" ]; then
    echo "missing checkpoint: start=$START base=$BASE" >&2
    exit 2
fi
if [ $((ENVS % 2)) -ne 0 ]; then
    echo "MSTOP_ENVS must be divisible by 2" >&2
    exit 2
fi
PER_GPU=$((ENVS / 2))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH="$ROOT/ddp_shim:${PYTHONPATH:-}"

CFG_DIR=output/mastop_carry/configs
CFG=$CFG_DIR/${TAG}.yaml
TRAIN_CFG=$CFG_DIR/${TAG}_train.yaml
mkdir -p "$CFG_DIR" output/mastop_carry/"$TAG" "$ROOT/runs/results/mastop_carry"
sed -e "s/^  numAgents:.*/  numAgents: 2/"     -e "s/^  numEnvs:.*/  numEnvs: $PER_GPU/"     -e "s/^  envSpacing:.*/  envSpacing: 5/"     -e "/default_buffer_size_multiplier:/i\\    max_gpu_contact_pairs: 4194304"     tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml > "$CFG"
sed -e "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MB/"     -e "s/^\( *\)amp_minibatch_size:.*/\1amp_minibatch_size: $AMP_MB/"     tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml > "$TRAIN_CFG"

export MSTOP_STAGE="$STAGE"
export MSTOP_MIX=${MSTOP_MIX:-}
[ -z "$MSTOP_MIX" ] && unset MSTOP_MIX
export MSTOP_SEED="$SEED"
export MS_SEED="$SEED"
export MS_MRAND=0
export MS_VEL_W=1
export MS_CLIP=0
export MA_SEP=0
export MA_SPAWN_GAP=1.0
export MA_C=0
export MA_BETA=0
export MA_TOKEN=live
export MSTOP_BRAKE_T=${MSTOP_BRAKE_T:-0.8}
export MA_TOKENIZER_ZERO=1
export MA_METRICS="$ROOT/runs/results/mastop_carry/${TAG}.npy"

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate "${MSTOP_CONDA_ENV:-tokenhsi_koo}"
BASE_EPOCH=$(python -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu').get('epoch',0))" "$START")
MAX_EPOCH=$((BASE_EPOCH + ITERS))
echo "MSTOP_TRAIN_START stage=$STAGE tag=$TAG GPUs=6,7 envs=$ENVS start_epoch=$BASE_EPOCH max_epoch=$MAX_EPOCH"
torchrun --standalone --nproc_per_node=2 ./tokenhsi/run.py     --task HumanoidMAStopCarry     --cfg_train "$TRAIN_CFG"     --cfg_env "$CFG"     --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml     --num_envs "$PER_GPU" --headless --seed "$SEED"     --hrl_checkpoint "$BASE" --checkpoint "$START" --resume 1     --max_iterations "$MAX_EPOCH"     --output_path "output/mastop_carry/$TAG" --horovod
CKPT=$(find "output/mastop_carry/$TAG" -name Humanoid.pth -printf "%T@ %p\n" | sort -rn | head -1 | cut -d" " -f2-)
test -n "$CKPT"
echo "MSTOP_TRAIN_DONE stage=$STAGE tag=$TAG checkpoint=$REPO/$CKPT"
