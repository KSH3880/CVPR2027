#!/bin/bash
# Usage: launch.sh <new-tag> <train|policy|nominal|rule|probe> [executor.pth] [speed-policy.pth]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: launch.sh new-tag mode [executor.pth] [speed-policy.pth]}
MODE=${2:?usage: launch.sh new-tag mode [executor.pth] [speed-policy.pth]}
EXECUTOR=${3:-${CARRY_SPEED_EXECUTOR:-/home/visitor/koo_cvpr/TokenHSI-masteer/output/masteer/ms18_carry_steer50_s0/Humanoid_23-12-39-56/nn/Humanoid_00012000.pth}}
STAGE1=${MS_CKPT:-/home/visitor/koo_cvpr/TokenHSI-masteer/output/tokenhsi/ckpt_stage1.pth}
POLICY=${4:-${CARRY_SPEED_INIT:-}}
GPU=${MA_GPU:-0}
ENVS=${CARRY_SPEED_ENVS:-256}
STEPS=${CARRY_SPEED_EPISODE_STEPS:-360}
SEED=${CARRY_SPEED_SEED:-0}
case "$TAG" in ""|*[!A-Za-z0-9_.-]*) echo "invalid tag" >&2; exit 2;; esac
case "$MODE" in train|policy|nominal|rule|probe) ;; *) echo "invalid mode: $MODE" >&2; exit 2;; esac
for file in "$EXECUTOR" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint missing; pass executor path and MS_CKPT explicitly: $file" >&2; exit 2; }
done
if [ "$MODE" = policy ] && [ -z "$POLICY" ]; then echo "policy mode requires a speed-policy checkpoint" >&2; exit 2; fi
if [ -n "$POLICY" ]; then [ -f "$POLICY" ] || { echo "speed policy missing: $POLICY" >&2; exit 2; }; POLICY=$(realpath -- "$POLICY"); fi
OUT="$ROOT/runs/carry_speed/$TAG"
[ ! -e "$OUT" ] || { echo "existing run: $OUT; use a new tag" >&2; exit 3; }
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
case "$UUID" in GPU-*|MIG-*) ;; *) echo "invalid GPU UUID" >&2; exit 2;; esac
if [ -z "${CONDA_BASE:-}" ]; then CONDA_BASE=/home/injesus1010/anaconda3; fi
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"
set -u
EXECUTOR=$(realpath -- "$EXECUTOR")
STAGE1=$(realpath -- "$STAGE1")
mkdir -p "$OUT"
python "$ROOT/TokenHSI-coord/carry_speed/configure.py" \
    "$ROOT/TokenHSI-coord/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" \
    "$OUT/env.yaml" --envs "$ENVS" --steps "$STEPS"
export CARRY_SPEED_OUTPUT="$OUT" CARRY_SPEED_MODE="$MODE" CARRY_SPEED_INIT="$POLICY"
export CARRY_PLANNER_PHYSICAL_GPU="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID"
export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0 COORD_VIEWER=0
export CARRY_PLANNER_CONVERGE_PROB=0 CARRY_PLANNER_RANDOMIZE_AGENT_SLOTS=1
export MA_TOKEN=mask MA_TOKENIZER_ZERO=1 MA_SEP=0 MA_C=0 MA_BETA=0
export MS_MRAND=0 MS_M_LO=0 MS_CLIP=1 MS_ZERO=0 MS_SCEN=cross MS_SEED="$SEED"
export MS_REWARD_OUTER=1 MS_POS_C=0.6 MS_VEL_W=1 MS_VIEW_TIMED_CROSS=0
export COORD_SPEED_LIMIT=1 COORD_ACCEL_UP=${COORD_ACCEL_UP:-0.75} COORD_ACCEL_DOWN=${COORD_ACCEL_DOWN:-1.0}
export COORD_PRESERVE_PICKUP_APPROACH=0 COORD_ALLOW_HAND_CONTACT=1
unset MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_S MA_LAYOUT_D MS_SCEN_CURVE
source "$ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"
{
    printf 'executor=%q\nstage1=%q\n' "$EXECUTOR" "$STAGE1"
    env | LC_ALL=C sort | rg '^(CARRY_SPEED_|CARRY_PLANNER_|COORD_|MA_|MS_|CUDA_VISIBLE_DEVICES=)'
} > "$OUT/run.env"
cd "$ROOT/TokenHSI-coord"
python -u -m carry_speed.run --test --headless --task HumanoidMACarrySpeedTrain \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id -1 --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$OUT/env.yaml" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$EXECUTOR" \
    --num_envs "$ENVS" --seed "$SEED" --output_path "$OUT/executor_unused" \
    > "$OUT/run.log" 2>&1
