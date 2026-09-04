#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
ROOT=$(cd "$REPO/.." && pwd)
cd "$REPO"

STAGE=${MSTOP_STAGE:-B}
case "$STAGE" in B) ;; *) echo "stop train stage must be B" >&2; exit 2;; esac
TAG=${MSTOP_TAG:-mastop_b_residual_s0}
INIT_MODE=${MSTOP_INIT_MODE:-resume}
START=${MSTOP_START_CKPT:-}
BASE=${MSTOP_BASE_CKPT:-$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth}
ITERS=${MSTOP_ITERS:-3000}
ENVS=${MSTOP_ENVS:-2048}
GPU=${MSTOP_GPU:-6,7}
SEED=${MSTOP_SEED:-0}
MB=${MSTOP_MB:-8192}
AMP_MB=${MSTOP_AMP_MB:-2048}
case "$INIT_MODE" in
    resume) test -f "$START" || { echo "missing start checkpoint: $START" >&2; exit 2; } ;;
    stage1) START= ;;
    *) echo "MSTOP_INIT_MODE must be resume or stage1" >&2; exit 2 ;;
esac
export MSTOP_INIT_MODE="$INIT_MODE"
test -f "$BASE" || { echo "missing base checkpoint: $BASE" >&2; exit 2; }
case "$GPU" in
    6|7) NPROC=1; PER_GPU=$ENVS ;;
    6,7)
        if [ $((ENVS % 2)) -ne 0 ]; then
            echo "MSTOP_ENVS must be divisible by 2 for DDP" >&2
            exit 2
        fi
        NPROC=2
        PER_GPU=$((ENVS / 2))
        ;;
    *) echo "MSTOP_GPU must be 6, 7, or 6,7" >&2; exit 2 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
if [ "$NPROC" -eq 2 ]; then
    export PYTHONPATH="$ROOT/ddp_shim:${PYTHONPATH:-}"
fi

CFG_DIR=output/mastop_carry/configs
CFG=$CFG_DIR/${TAG}.yaml
TRAIN_CFG=$CFG_DIR/${TAG}_train.yaml
OUT=output/mastop_carry/$TAG
ENV_FILE=$ROOT/runs/logs/mastop_carry/$TAG.env
if [ -e "$OUT" ] || [ -e "$ENV_FILE" ]; then
    echo "refusing to reuse existing tag: $TAG" >&2
    exit 3
fi
mkdir -p "$CFG_DIR" "$OUT" "$ROOT/runs/results/mastop_carry" "$(dirname "$ENV_FILE")"
sed -e "s/^  numAgents:.*/  numAgents: 2/"     -e "s/^  numEnvs:.*/  numEnvs: $PER_GPU/"     -e "s/^  envSpacing:.*/  envSpacing: 5/"     -e "/default_buffer_size_multiplier:/i\\    max_gpu_contact_pairs: 4194304"     tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml > "$CFG"
sed -e "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MB/"     -e "s/^\( *\)amp_minibatch_size:.*/\1amp_minibatch_size: $AMP_MB/"     tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml > "$TRAIN_CFG"

export MSTOP_STAGE="$STAGE"
export MSTOP_MIX=${MSTOP_MIX:-0,1,0,0}
export MSTOP_SEED="$SEED"
export MS_SEED="$SEED"
export MS_MRAND=0
export MS_VEL_W=1
export MS_VEL_K=5
export MS_POS_C=2
export MS_ENDCLAMP=0
export MS_CLIP=1
export MA_SEP=9
export MA_SPAWN_GAP=1.0
export MA_C=0
export MA_BETA=0
export MA_TOKEN=zero
POLICY_MODE=${MSTOP_POLICY_MODE:-residual}
export MSTOP_POLICY_MODE="$POLICY_MODE"
case "$POLICY_MODE" in
    residual)
        export MA_OLD_CARRY_ONLY=1 MA_TEAMMATE_MASK=1
        export MSTOP_RESIDUAL_ARCH=1 MSTOP_RESIDUAL=1
        export MSTOP_STOP_HEAD_ARCH=0 MSTOP_STOP_HEAD=0 MSTOP_STOP_HEAD_WARMSTART=0
        export MSTOP_FROZEN_A=1
        ;;
    stop_head)
        export MA_OLD_CARRY_ONLY=1 MA_TEAMMATE_MASK=1
        export MSTOP_RESIDUAL_ARCH=0 MSTOP_RESIDUAL=0
        export MSTOP_STOP_HEAD_ARCH=1 MSTOP_STOP_HEAD=1 MSTOP_STOP_HEAD_WARMSTART=1
        export MSTOP_FROZEN_A=1
        ;;
    joint_stop_head)
        export MA_OLD_CARRY_ONLY=0 MA_TEAMMATE_MASK=0
        export MSTOP_RESIDUAL_ARCH=0 MSTOP_RESIDUAL=0
        export MSTOP_STOP_HEAD_ARCH=1 MSTOP_STOP_HEAD=1 MSTOP_STOP_HEAD_WARMSTART=0
        export MSTOP_FROZEN_A=0
        ;;
    *) echo "MSTOP_POLICY_MODE must be residual, stop_head, or joint_stop_head" >&2; exit 2 ;;
esac
export MSTOP_RESIDUAL_SCALE=${MSTOP_RESIDUAL_SCALE:-2}
export MSTOP_BRAKE_T=${MSTOP_BRAKE_T:-0.8}
export MSTOP_RESUME_T=${MSTOP_RESUME_T:-0.8}
export MSTOP_STOP_SPEED=${MSTOP_STOP_SPEED:-0.2}
export MSTOP_TRACK_W=${MSTOP_TRACK_W:-0.5}
export MSTOP_TRACK_DELTA=${MSTOP_TRACK_DELTA:-0.5}
export MSTOP_STOP_PROB=${MSTOP_STOP_PROB:-1}
export MSTOP_HOLD_STRICT_W=${MSTOP_HOLD_STRICT_W:-0}
export MSTOP_HOLD_CONTACT_W=${MSTOP_HOLD_CONTACT_W:-0}
export MSTOP_POST_DROP_C=${MSTOP_POST_DROP_C:-0}
export MSTOP_HOLD_SLIP_C=${MSTOP_HOLD_SLIP_C:-0}
export MSTOP_HOLD_LAT_C=${MSTOP_HOLD_LAT_C:-0}
export MSTOP_HOLD_YAW_C=${MSTOP_HOLD_YAW_C:-0}
export MSTOP_CARRYWITH=${MSTOP_CARRYWITH:-1}
export MSTOP_ARM_SCALE=${MSTOP_ARM_SCALE:-1}
export MSTOP_RESUME_RESIDUAL_SCALE=${MSTOP_RESUME_RESIDUAL_SCALE:-1}
export MSTOP_DIAG=${MSTOP_DIAG:-1}
export MSTOP_DIAG_EVERY=${MSTOP_DIAG_EVERY:-10}
export MSTOP_DIAG_PATH=${MSTOP_DIAG_PATH:-$ROOT/runs/results/mastop_carry/${TAG}_diag.jsonl}
export MSTOP_PHASE_REWARD=${MSTOP_PHASE_REWARD:-0}
export MS_GRADCHK=${MS_GRADCHK:-1}
export MA_TOKENIZER_ZERO=1
export MA_METRICS="$ROOT/runs/results/mastop_carry/${TAG}.npy"
{
    for name in MSTOP_STAGE MSTOP_MIX MSTOP_SEED MSTOP_STOP_PROB MSTOP_BRAKE_T MSTOP_RESUME_T MSTOP_STOP_SPEED \
        MSTOP_TRACK_W MSTOP_TRACK_DELTA \
        MSTOP_HOLD_STRICT_W MSTOP_HOLD_CONTACT_W MSTOP_POST_DROP_C \
        MSTOP_HOLD_SLIP_C MSTOP_HOLD_LAT_C MSTOP_HOLD_YAW_C \
        MSTOP_CARRYWITH MSTOP_RESIDUAL_ARCH \
        MSTOP_RESIDUAL MSTOP_STOP_HEAD_ARCH MSTOP_STOP_HEAD \
        MSTOP_STOP_HEAD_WARMSTART MSTOP_FROZEN_A MSTOP_RESIDUAL_SCALE \
        MSTOP_ARM_SCALE MSTOP_RESUME_RESIDUAL_SCALE \
        MSTOP_DIAG MSTOP_DIAG_EVERY MSTOP_DIAG_PATH \
        MSTOP_PHASE_REWARD MSTOP_INIT_MODE MSTOP_POLICY_MODE \
        MA_OLD_CARRY_ONLY MA_TEAMMATE_MASK \
        MA_TOKEN MA_TOKENIZER_ZERO MA_SEP MA_SPAWN_GAP MA_C MA_BETA \
        MS_MRAND MS_VEL_W MS_VEL_K MS_POS_C MS_CLIP MS_ENDCLAMP MS_SEED; do
        printf 'export %s=%q\n' "$name" "${!name}"
    done
    printf 'export MSTOP_START_CKPT=%q\n' "$START"
    printf 'export MSTOP_BASE_CKPT=%q\n' "$BASE"
    printf 'export MSTOP_ITERS=%q\n' "$ITERS"
    printf 'export MSTOP_ENVS=%q\n' "$ENVS"
    printf 'export MSTOP_ENVS_PER_GPU=%q\n' "$PER_GPU"
    printf 'export MSTOP_GPU=%q\n' "$GPU"
} > "$ENV_FILE"

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate "${MSTOP_CONDA_ENV:-tokenhsi_koo}"
BASE_EPOCH=0
LOAD=()
if [ "$INIT_MODE" = resume ]; then
    BASE_EPOCH=$(python -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu').get('epoch',0))" "$START")
    LOAD=(--checkpoint "$START" --resume 1)
fi
MAX_EPOCH=$((BASE_EPOCH + ITERS))
if [ "$NPROC" -eq 2 ]; then
    RUN=(torchrun --standalone --nproc_per_node=2 ./tokenhsi/run.py)
    MULTI=(--horovod)
else
    RUN=(python -u ./tokenhsi/run.py)
    MULTI=()
fi
echo "MSTOP_TRAIN_START stage=$STAGE tag=$TAG GPUs=$GPU total_envs=$ENVS per_gpu=$PER_GPU start_epoch=$BASE_EPOCH max_epoch=$MAX_EPOCH"
"${RUN[@]}" \
    --task HumanoidMAStopCarry \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs "$PER_GPU" --headless --seed "$SEED" \
    --hrl_checkpoint "$BASE" "${LOAD[@]}" \
    --max_iterations "$MAX_EPOCH" --output_path "$OUT" "${MULTI[@]}"
CKPT=$(find "$OUT" -name Humanoid.pth -printf "%T@ %p\n" | sort -rn | head -1 | cut -d" " -f2-)
test -n "$CKPT"
echo "MSTOP_TRAIN_DONE stage=$STAGE tag=$TAG checkpoint=$REPO/$CKPT"
