#!/bin/bash
# Fine-tune only new_carry tokenizer + internal action residual from a trained
# MA-steer checkpoint.  The teammate token is kept in the 340-D ABI but is
# masked from transformer attention.
#
# Smoke:
#   MA_GPU=0 STACK_ENVS=16 STACK_ITERS=2 MS_GRADCHK=1 \
#     bash scripts/masteer/train_sequential_stack.sh seq_smoke
#
# Short run:
#   MA_GPU=0 STACK_ENVS=256 STACK_ITERS=500 \
#     bash scripts/masteer/train_sequential_stack.sh seq_ft_s0
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
TAG=${1:?usage: train_sequential_stack.sh <tag> [ma-steer-policy.pth]}
POLICY=${2:-"$REPO/output/stack/ms18_maskteam_origscale_c06_s0_00009000.pth"}
STAGE1=${MS_CKPT:-"$REPO/output/ckpt_stage1.pth"}
ENVS=${STACK_ENVS:-2048}
ITERS=${STACK_ITERS:-2}
SEED=${STACK_SEED:-0}
# Same device contract as scripts/f22/train_stack.sh: physical GPU 7 by
# default, remapped through CUDA_VISIBLE_DEVICES to logical cuda:0.
GPU=${MA_GPU:-7}
SAVE_FREQ=${STACK_SAVE_FREQ:-100}
OUT="output/sequential_stack/$TAG"
HORIZON=32

case "$TAG" in
    ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2 ;;
esac
case "$GPU" in
    0|1|2|3|4|5|6|7) ;;
    *) echo "MA_GPU는 physical GPU index 0..7이어야 한다: $GPU" >&2; exit 2 ;;
esac
for file in "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 1; }
done
case "$ENVS:$ITERS:$SAVE_FREQ" in
    *[!0-9:]*|:*|*:) echo "STACK_ENVS/ITERS/SAVE_FREQ는 양의 정수여야 한다" >&2; exit 2 ;;
esac
[ "$ENVS" -gt 0 ] && [ "$ITERS" -gt 0 ] && [ "$SAVE_FREQ" -gt 0 ] || exit 2
[ ! -e "$REPO/$OUT" ] || {
    echo "기존 output tag를 덮어쓰지 않는다: $REPO/$OUT" >&2
    exit 1
}

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    else
        CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || {
    echo "conda 초기화 파일 없음: $CONDA_BASE/etc/profile.d/conda.sh" >&2
    exit 1
}
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"

BATCH_SIZE=$((ENVS * HORIZON * 2))
MINIBATCH=${STACK_MINIBATCH_SIZE:-16384}
if [ "$MINIBATCH" -gt "$BATCH_SIZE" ] || [ $((BATCH_SIZE % MINIBATCH)) -ne 0 ]; then
    MINIBATCH=1
    while [ $((MINIBATCH * 2)) -le 16384 ] \
        && [ $((MINIBATCH * 2)) -le "$BATCH_SIZE" ] \
        && [ $((BATCH_SIZE % (MINIBATCH * 2))) -eq 0 ]; do
        MINIBATCH=$((MINIBATCH * 2))
    done
fi

BASE_EPOCH=$(python - "$POLICY" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
rms = c["running_mean_std"]["running_mean"].numel()
if rms != 340:
    raise SystemExit(f"MA-steer policy RMS must be 340-D, got {rms}: {sys.argv[1]}")
print(int(c.get("epoch", 0)))
PY
)
FINAL_EPOCH=$((BASE_EPOCH + ITERS))

CFG=$(mktemp /tmp/sequential_stack_env.XXXXXX.yaml)
TRAIN_CFG=$(mktemp /tmp/sequential_stack_train.XXXXXX.yaml)
cleanup() {
    rm -f -- "$CFG" "$TRAIN_CFG"
}
trap cleanup EXIT INT TERM

sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    "$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" \
    > "$CFG"
sed -e "s/^\( *\)save_frequency:.*/\1save_frequency: $SAVE_FREQ/" \
    -e 's/^\( *\)save_intermediate:.*/\1save_intermediate: True/' \
    -e "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MINIBATCH/" \
    "$REPO/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
    > "$TRAIN_CFG"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
unset TOKENHSI_GRAPHICS_DEVICE_ID
export MA_TOKEN=mask
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_FINETUNE_NEWCARRY_RESIDUAL=1
export MA_NOFREEZE=0
export MS_GRADCHK=${MS_GRADCHK:-1}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=${MS_CLIP:-1}
# Exact ms18 reward contract.  Do not fall back to the older masteer defaults
# (OUTER=2, POS_C=2), which would change the policy's reward scale at resume.
export MS_REWARD_OUTER=${MS_REWARD_OUTER:-1}
export MS_POS_C=${MS_POS_C:-0.6}
export MS_VEL_W=${MS_VEL_W:-1}
export MS_SCEN=${MS_SCEN:-free}
export STACK_TASK_MODE=stack
export STACK_HAND_CLEAR_START=${STACK_HAND_CLEAR_START:-0.08}
export STACK_HAND_CLEAR_DONE=${STACK_HAND_CLEAR_DONE:-0.20}
export STACK_RELEASE_REWARD_W=${STACK_RELEASE_REWARD_W:-0.50}
export STACK_EARLY_RELEASE_PENALTY=${STACK_EARLY_RELEASE_PENALTY:-0.50}

PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-/tmp/hwanhee-physx-lib}
if [ -d "$PHYSX_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

echo "=============================================================="
echo " task       HumanoidMASequentialStackCarry"
echo " policy     $POLICY (epoch $BASE_EPOCH)"
echo " stage1     $STAGE1"
echo " trainable  new_carry tokenizer + internal residual"
echo " masked     teammate token"
echo " frozen     self/teammate/steer/old-carry/transformer/composer/RMS"
echo " ms18 rwd   OUTER=$MS_REWARD_OUTER POS_C=$MS_POS_C VEL_W=$MS_VEL_W"
echo " release    clear=${STACK_HAND_CLEAR_START}..${STACK_HAND_CLEAR_DONE}m reward=$STACK_RELEASE_REWARD_W early_pen=$STACK_EARLY_RELEASE_PENALTY"
echo " envs       $ENVS x 2 agents"
echo " PPO batch  $BATCH_SIZE (minibatch $MINIBATCH)"
echo " iterations +$ITERS -> final epoch $FINAL_EPOCH"
echo " GPU        physical $GPU -> logical cuda:0"
echo " output     $REPO/$OUT"
echo "=============================================================="

cd "$REPO"
python -u ./tokenhsi/run.py \
    --task HumanoidMASequentialStackCarry \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id -1 \
    --physx --pipeline gpu --headless \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" \
    --checkpoint "$POLICY" --resume 1 \
    --num_envs "$ENVS" --max_iterations "$FINAL_EPOCH" \
    --output_path "$OUT" --seed "$SEED"
