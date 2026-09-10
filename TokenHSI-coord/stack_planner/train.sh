#!/bin/bash
# Isolated stack-planner PPO; the sequential-stack executor stays frozen.
# Smoke: STACK_PLANNER_ENVS=8 STACK_PLANNER_ITERS=2 STACK_PLANNER_HORIZON=4 \
#   bash TokenHSI-coord/stack_planner/train.sh smoke_s0
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD="$ROOT/TokenHSI-coord"
EXEC_REPO="$ROOT/TokenHSI-masteer"
TAG=${1:?usage: train.sh <tag> [sequential-stack-policy.pth]}
EXEC_CKPT=${2:-"$EXEC_REPO/output/sequential_stack/anti_feat_top_s2/Humanoid_00011000.pth"}
STAGE1=${MS_CKPT:-"$EXEC_REPO/output/ckpt_stage1.pth"}
ENVS=${STACK_PLANNER_ENVS:-64}
GPU=${MA_GPU:-0}
SEED=${STACK_PLANNER_SEED:-0}
OUT="$ROOT/runs/stack_planner/$TAG"

case "$TAG" in ""|*[!A-Za-z0-9_.-]*) echo "invalid tag: $TAG" >&2; exit 2;; esac
case "$ENVS:$GPU" in *[!0-9:]*) echo "ENVS/GPU must be non-negative integers" >&2; exit 2;; esac
[ "$ENVS" -gt 0 ] || exit 2
for file in "$EXEC_CKPT" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 1; }
done
[ ! -e "$OUT" ] || { echo "기존 run을 덮어쓰지 않는다: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$ROOT/runs/gen_cfgs/stack_planner"

CFG="$ROOT/runs/gen_cfgs/stack_planner/$TAG.yaml"
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    "$EXEC_REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"

if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    else CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export STACK_PLANNER_OUTPUT="$OUT"
export STACK_PLANNER_ITERS=${STACK_PLANNER_ITERS:-200}
export STACK_PLANNER_HORIZON=${STACK_PLANNER_HORIZON:-32}
export STACK_PLANNER_LOW_STEPS=${STACK_PLANNER_LOW_STEPS:-6}
export STACK_PLANNER_PPO_EPOCHS=${STACK_PLANNER_PPO_EPOCHS:-3}
export STACK_PLANNER_MINIBATCH=${STACK_PLANNER_MINIBATCH:-512}
export STACK_PLANNER_SAVE_EVERY=${STACK_PLANNER_SAVE_EVERY:-10}
export STACK_PLANNER_LR=${STACK_PLANNER_LR:-0.0003}

export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1 MS_ZERO=0 MS_REWARD_OUTER=1 MS_POS_C=${MS_POS_C:-1.2}
export MS_VEL_W=1 MS_SCEN=${MS_SCEN:-free}
export STACK_TASK_MODE=stack STACK_ALLOW_HAND_CONTACT=1
export STACK_CARRY_REHEARSAL_PROB=0 STACK_END_ON_A2_RESUME=0
export STACK_VIRTUAL_RETREAT_BOX=1 STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-900}
export STACK_BOTTOM_Z_TOL=${STACK_BOTTOM_Z_TOL:-0.05}
export STACK_BOTTOM_DISPLACE_TOL=${STACK_BOTTOM_DISPLACE_TOL:-0.50}
export STACK_TOP_XY_TOL=${STACK_TOP_XY_TOL:-0.15}
export STACK_TOP_FOLLOWS_BOTTOM=${STACK_TOP_FOLLOWS_BOTTOM:-1}

PHYSX_LIB_DIR=${PHYSX_LIB_DIR:-/tmp/hwanhee-physx-lib}
if [ -d "$PHYSX_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$PHYSX_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

{
    printf 'executor=%q\n' "$(realpath -- "$EXEC_CKPT")"
    printf 'stage1=%q\n' "$(realpath -- "$STAGE1")"
    printf 'cfg=%q\n' "$CFG"
    for name in CUDA_VISIBLE_DEVICES STACK_PLANNER_OUTPUT STACK_PLANNER_ITERS \
        STACK_PLANNER_HORIZON STACK_PLANNER_LOW_STEPS STACK_PLANNER_PPO_EPOCHS \
        STACK_PLANNER_MINIBATCH STACK_PLANNER_SAVE_EVERY STACK_PLANNER_LR \
        MA_TOKEN MA_TOKENIZER_ZERO MS_MRAND MS_M_LO MS_CLIP MS_ZERO MS_SCEN \
        STACK_TASK_MODE STACK_CARRY_REHEARSAL_PROB STACK_VIRTUAL_RETREAT_BOX \
        STACK_TOP_FOLLOWS_BOTTOM; do
        printf '%s=%q\n' "$name" "${!name}"
    done
} > "$OUT/run.env"

echo "stack planner: tag=$TAG envs=$ENVS gpu=$GPU frozen=$EXEC_CKPT"
cd "$COORD"
python -u -m stack_planner.train_closed_loop \
    --test --headless --task HumanoidMAStackPlannerTrain \
    --cfg_train "$EXEC_REPO/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
    --cfg_env "$CFG" \
    --motion_file "$EXEC_REPO/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
    --hrl_checkpoint "$STAGE1" --checkpoint "$EXEC_CKPT" \
    --num_envs "$ENVS" --seed "$SEED" --output_path "$OUT/executor_unused"
