#!/bin/bash
# Plain simultaneous Carry collision-avoidance planner. The ms18 executor is frozen.
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD="$ROOT/TokenHSI-coord"
EXEC_REPO=${CARRY_PLANNER_EXEC_REPO:-"$ROOT/TokenHSI-masteer"}
TAG=${1:?usage: train.sh <tag> <ms18-policy.pth>}
EXEC_CKPT=${2:?usage: train.sh <tag> <ms18-policy.pth>}
STAGE1=${MS_CKPT:-"/home/hwanhee/juan/CVPR2027/TokenHSI-masteer/output/tokenhsi/ckpt_stage1.pth"}
ENVS=${CARRY_PLANNER_ENVS:-2048}
GPU=${MA_GPU:-7}
SEED=${CARRY_PLANNER_SEED:-0}
OUT="$ROOT/runs/carry_planner/$TAG"

case "$TAG" in ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2;; esac
case "$ENVS:$GPU" in *[!0-9:]*) echo "ENVS/GPU must be non-negative integers" >&2; exit 2;; esac
[ "$ENVS" -gt 0 ] || exit 2
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null) || {
    echo "NVML physical GPU index $GPU is not available" >&2; exit 2;
}
case "$GPU_UUID" in GPU-*|MIG-*) ;; *) echo "invalid GPU UUID: $GPU_UUID" >&2; exit 2;; esac
for file in "$EXEC_CKPT" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 1; }
done
EXEC_CKPT=$(realpath -- "$EXEC_CKPT")
STAGE1=$(realpath -- "$STAGE1")
[ ! -e "$OUT" ] || { echo "기존 run을 덮어쓰지 않는다: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$ROOT/runs/gen_cfgs/carry_planner"

CFG="$ROOT/runs/gen_cfgs/carry_planner/$TAG.yaml"
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    "/home/hwanhee/juan/CVPR2027/TokenHSI-masteer/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    else CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi_juan}"

# MA_GPU is a nvitop/NVML physical index. CUDA numeric ordinals may have a
# different order on multi-GPU servers, so select the exact device by UUID.
export CARRY_PLANNER_PHYSICAL_GPU="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID"
export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0
export CARRY_PLANNER_OUTPUT="$OUT"
export CARRY_PLANNER_ITERS=${CARRY_PLANNER_ITERS:-200}
export CARRY_PLANNER_HORIZON=${CARRY_PLANNER_HORIZON:-32}
export CARRY_PLANNER_LOW_STEPS=${CARRY_PLANNER_LOW_STEPS:-6}
export CARRY_PLANNER_HISTORY_STEPS=${CARRY_PLANNER_HISTORY_STEPS:-4}
export CARRY_PLANNER_COLLISION_COEF=${CARRY_PLANNER_COLLISION_COEF:-5.0}
export CARRY_PLANNER_CONVERGE_PROB=${CARRY_PLANNER_CONVERGE_PROB:-0.75}
export CARRY_PLANNER_GOAL_MARGIN=${CARRY_PLANNER_GOAL_MARGIN:-0.25}
# Kept in the checkpoint config for shared stack-planner compatibility. Plain
# Carry uses the sparse control-point scale below instead of dense deltas.
export CARRY_PLANNER_DELTA_SCALE=${CARRY_PLANNER_DELTA_SCALE:-1.0}
export CARRY_PLANNER_CONTROL_SCALE=${CARRY_PLANNER_CONTROL_SCALE:-4.0}
# Exploration is applied to four sparse point logits per agent.
export CARRY_PLANNER_PATH_UPDATE_ALPHA=${CARRY_PLANNER_PATH_UPDATE_ALPHA:-0.5}
export CARRY_PLANNER_DELTA_STD=${CARRY_PLANNER_DELTA_STD:-0.25}
export CARRY_PLANNER_PPO_EPOCHS=${CARRY_PLANNER_PPO_EPOCHS:-3}
DEFAULT_MINIBATCH=$((ENVS * CARRY_PLANNER_HORIZON / 4))
[ "$DEFAULT_MINIBATCH" -gt 0 ] || DEFAULT_MINIBATCH=1
export CARRY_PLANNER_MINIBATCH=${CARRY_PLANNER_MINIBATCH:-$DEFAULT_MINIBATCH}
export CARRY_PLANNER_SAVE_EVERY=${CARRY_PLANNER_SAVE_EVERY:-5}
export CARRY_PLANNER_LR=${CARRY_PLANNER_LR:-0.0003}
export CARRY_PLANNER_GAMMA=${CARRY_PLANNER_GAMMA:-0.99}
export CARRY_PLANNER_CLIP=${CARRY_PLANNER_CLIP:-0.2}
export CARRY_PLANNER_VALUE_COEF=${CARRY_PLANNER_VALUE_COEF:-0.5}
export CARRY_PLANNER_ENTROPY_COEF=${CARRY_PLANNER_ENTROPY_COEF:-0.0001}
export CARRY_PLANNER_SMOOTHNESS_COEF=${CARRY_PLANNER_SMOOTHNESS_COEF:-10.0}
export CARRY_PLANNER_SPEED_SMOOTHNESS_COEF=${CARRY_PLANNER_SPEED_SMOOTHNESS_COEF:-1.0}
export CARRY_PLANNER_ANALYTIC_COLLISION_COEF=${CARRY_PLANNER_ANALYTIC_COLLISION_COEF:-1.0}
export CARRY_PLANNER_ANALYTIC_CURVATURE_COEF=${CARRY_PLANNER_ANALYTIC_CURVATURE_COEF:-20.0}
export CARRY_PLANNER_ANALYTIC_FOCUS_STEPS=${CARRY_PLANNER_ANALYTIC_FOCUS_STEPS:-8}
export CARRY_PLANNER_INVALID_PLAN_COEF=${CARRY_PLANNER_INVALID_PLAN_COEF:-0.25}
export CARRY_PLANNER_INIT=${CARRY_PLANNER_INIT:-}
if [ -n "$CARRY_PLANNER_INIT" ]; then
    [ -f "$CARRY_PLANNER_INIT" ] || {
        echo "planner init checkpoint 없음: $CARRY_PLANNER_INIT" >&2; exit 1;
    }
    CARRY_PLANNER_INIT=$(realpath -- "$CARRY_PLANNER_INIT")
    export CARRY_PLANNER_INIT
fi

export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_SEP=${MA_SEP:-0} MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1 MS_ZERO=0 MS_REWARD_OUTER=1 MS_POS_C=${MS_POS_C:-1.2}
export MS_VEL_W=1 MS_SCEN=${MS_SCEN:-cross}
export COORD_SPEED_LIMIT=1 COORD_ACCEL_UP=${COORD_ACCEL_UP:-0.75}
export COORD_ACCEL_DOWN=${COORD_ACCEL_DOWN:-1.0}
export COORD_ALLOW_HAND_CONTACT=${COORD_ALLOW_HAND_CONTACT:-1}
export COORD_PRESERVE_PICKUP_APPROACH=${COORD_PRESERVE_PICKUP_APPROACH:-1}

. "$COORD/stack_planner/physx_cuda_compat.sh"

{
    printf 'executor=%q\n' "$(realpath -- "$EXEC_CKPT")"
    printf 'stage1=%q\n' "$(realpath -- "$STAGE1")"
    printf 'cfg=%q\n' "$CFG"
    env | LC_ALL=C sort | grep -E '^(CARRY_PLANNER_|COORD_|MA_|MS_|CUDA_VISIBLE_DEVICES=)'
} > "$OUT/run.env"

echo "carry planner: tag=$TAG envs=$ENVS physical_gpu=$GPU uuid=$GPU_UUID logical_gpu=0 scene=$MS_SCEN frozen=$EXEC_CKPT"
cd "$COORD"
python -u -m carry_planner.train_closed_loop \
    --test --headless --task HumanoidMACarryPlannerTrain \
    --cfg_train /home/hwanhee/juan/CVPR2027/TokenHSI-masteer/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file /home/hwanhee/juan/CVPR2027/TokenHSI-masteer/tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$EXEC_CKPT" \
    --num_envs "$ENVS" --seed "$SEED" --output_path "$OUT/executor_unused"
