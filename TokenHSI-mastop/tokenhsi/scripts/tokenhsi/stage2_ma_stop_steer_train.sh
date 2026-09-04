#!/bin/bash
# One Stage-2 run with episode-level free/solo/near/conflict mixing.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"

TAG=${MSTOP_TAG:-mastop_mixed_s0}
GPUS=${MSTOP_GPUS:-6,7}
case "$GPUS" in
    6,7) ;;
    *) echo "MSTOP_GPUS must be exactly 6,7" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPUS
NPROC=2
ENVS=${MSTOP_ENVS:-2048}
if [ $((ENVS % NPROC)) -ne 0 ]; then
    echo "MSTOP_ENVS=$ENVS must be divisible by $NPROC GPUs" >&2
    exit 2
fi
PER_GPU_ENVS=$((ENVS / NPROC))
ITERS=${MSTOP_ITERS:-15000}
SEED=${MSTOP_SEED:-0}
MB=${MSTOP_MB:-8192}
AMP_MB=${MSTOP_AMP_MB:-2048}
DEFAULT_BASE_CKPT="$(dirname "$REPO")/TokenHSI/output/tokenhsi/ckpt_stage1.pth"
BASE_CKPT=${MSTOP_BASE_CKPT:-$DEFAULT_BASE_CKPT}
CFG_SRC=tokenhsi/data/cfg/mastop/amp_humanoid_ma_stop_steer_mixed.yaml
CFG=output/mastop/configs/${TAG}.yaml
TRAIN_CFG=${MSTOP_TRAIN_CFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml}

if [ ! -f "$BASE_CKPT" ]; then
    echo "missing Stage-1 checkpoint: $BASE_CKPT" >&2
    exit 2
fi

mkdir -p "$(dirname "$CFG")" output/mastop/metrics
sed "s/^  numEnvs:.*/  numEnvs: $PER_GPU_ENVS/" "$CFG_SRC" > "$CFG"
TRAIN_RUNTIME=output/mastop/configs/${TAG}_train.yaml
sed \
    -e "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MB/" \
    -e "s/^\( *\)amp_minibatch_size:.*/\1amp_minibatch_size: $AMP_MB/" \
    "$TRAIN_CFG" > "$TRAIN_RUNTIME"
TRAIN_CFG=$TRAIN_RUNTIME

# Same weights and observation shape are used throughout. Only the episode
# sampler changes the command distribution; there are no phase checkpoints.
export MSTOP_SCEN=${MSTOP_SCEN:-mixed}
export MSTOP_SEED=$SEED
export MSTOP_MIX=${MSTOP_MIX:-0.35,0.20,0.20,0.25}
export MSTOP_SPEED=${MSTOP_SPEED:-1.5}
export MSTOP_HORIZON=${MSTOP_HORIZON:-2.5}
export MSTOP_SAFE_D=${MSTOP_SAFE_D:-0.9}
export MSTOP_CLEAR=${MSTOP_CLEAR:-1.0}
export MSTOP_PARALLEL_GAP=${MSTOP_PARALLEL_GAP:-2.0}
export MSTOP_NEAR_DT_MIN=${MSTOP_NEAR_DT_MIN:-1.25}
export MSTOP_NEAR_DT_MAX=${MSTOP_NEAR_DT_MAX:-2.50}
export MSTOP_CONFLICT_DT_MAX=${MSTOP_CONFLICT_DT_MAX:-0.35}
export MSTOP_ANGLE_MIN=${MSTOP_ANGLE_MIN:-60}
export MSTOP_ANGLE_MAX=${MSTOP_ANGLE_MAX:-120}
export MSTOP_SPEED_EMA=${MSTOP_SPEED_EMA:-0.20}
export MSTOP_PATH_W=${MSTOP_PATH_W:-0.35}
export MSTOP_VEL_W=${MSTOP_VEL_W:-0.65}
export MSTOP_CREEP_C=${MSTOP_CREEP_C:-0.20}
export MSTOP_CREEP_V=${MSTOP_CREEP_V:-0.10}
export MSTOP_SOLO_RANDOM=1
export MSTOP_METRICS=${MSTOP_METRICS:-output/mastop/metrics/${TAG}.npy}

# --resume is required to load the Stage-1 actor. Add ITERS to its epoch so the
# run does not immediately terminate at the checkpoint's existing epoch.
BASE_EPOCH=$(python -c \
    "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu').get('epoch', 0))" \
    "$BASE_CKPT")
MAX_EPOCH=$((BASE_EPOCH + ITERS))
echo "[mastop] Stage-1 epoch=$BASE_EPOCH + $ITERS -> $MAX_EPOCH"
echo "[mastop] GPUs=$GPUS total_envs=$ENVS per_gpu=$PER_GPU_ENVS"
echo "[mastop] minibatch=$MB/rank amp_minibatch=$AMP_MB/rank"
echo "[mastop] mix=$MSTOP_MIX seed=$SEED metrics=$MSTOP_METRICS"

export PYTHONPATH="$(dirname "$REPO")/ddp_shim:${PYTHONPATH:-}"
torchrun --standalone --nproc_per_node=$NPROC ./tokenhsi/run.py \
    --task HumanoidMAStopSteer \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --checkpoint "$BASE_CKPT" --resume 1 \
    --max_iterations "$MAX_EPOCH" \
    --output_path "output/mastop/$TAG" \
    --num_envs "$PER_GPU_ENVS" --headless --seed "$SEED" \
    --horovod
