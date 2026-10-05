#!/bin/bash
# Usage: bash task_allocation/launch.sh new-tag [ms18.pth] [run|--dry-run]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: launch.sh new-tag [ms18.pth]}
case "$TAG" in ""|*[!A-Za-z0-9_.-]*) exit 2;; esac
MODE=${3:-run}
case "$MODE" in run|--dry-run) ;; *) echo "third argument: run or --dry-run" >&2; exit 2;; esac
CHECKPOINT=${2:-"$ROOT/TokenHSI-masteer/output/stack/ms18_maskteam_origscale_c06_s0_00009000.pth"}
STAGE1=${MS_CKPT:-"$ROOT/TokenHSI-masteer/output/tokenhsi/ckpt_stage1.pth"}
[ -f "$CHECKPOINT" ] && [ -f "$STAGE1" ] || { echo 'missing ms18/stage1 checkpoint' >&2; exit 2; }
OUT="$ROOT/runs/task_allocation/$TAG"
[ ! -e "$OUT" ] || { echo "use a new tag: $OUT" >&2; exit 3; }
GPU=${MA_GPU:-6}
[ "$GPU" = 6 ] || { echo "task allocation must use GPU 6" >&2; exit 2; }
ENVS=${ALLOC_ENVS:-64}
SEED=${ALLOC_SEED:-0}
EPISODE=${ALLOC_EPISODE_STEPS:-600}
export ALLOC_INTERVAL=${ALLOC_INTERVAL:-30} ALLOC_HORIZON=${ALLOC_HORIZON:-16} ALLOC_ITERS=${ALLOC_ITERS:-100}
export ALLOC_MODE=${ALLOC_MODE:-train} ALLOC_D_MODEL=${ALLOC_D_MODEL:-64} ALLOC_LR=${ALLOC_LR:-0.0003}
export ALLOC_EPOCHS=${ALLOC_EPOCHS:-4} ALLOC_MINIBATCH=${ALLOC_MINIBATCH:-256}
export ALLOC_GAMMA=${ALLOC_GAMMA:-0.999} ALLOC_LAMBDA=${ALLOC_LAMBDA:-0.995}
export ALLOC_TIME_COEF=${ALLOC_TIME_COEF:-1} ALLOC_DELIVERY_COEF=${ALLOC_DELIVERY_COEF:-10}
export ALLOC_FAILURE_COEF=${ALLOC_FAILURE_COEF:-40} ALLOC_SWITCH_COEF=${ALLOC_SWITCH_COEF:-0.1}
export ALLOC_EXTENT=${ALLOC_EXTENT:-3} ALLOC_CLEARANCE=${ALLOC_CLEARANCE:-1.2} ALLOC_SAVE_EVERY=${ALLOC_SAVE_EVERY:-10}
export ALLOC_ENVS="$ENVS" ALLOC_SEED="$SEED" ALLOC_EPISODE_STEPS="$EPISODE"
# Attach only to the existing GPU6 server; do not start/stop MPS here.
export MPS_GPU_ROOT=${MPS_GPU_ROOT:-/tmp/mps-test-${UID}}
source "$ROOT/mps/shell.sh"
mps_use "$GPU"
UUID="$TOKENHSI_GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "$MODE" = --dry-run ]; then
    printf 'tag=%s gpu=%s envs=%s iterations=%s interval=%s
' "$TAG" "$GPU" "$ENVS" "$ALLOC_ITERS" "$ALLOC_INTERVAL"
    printf 'checkpoint=%s
stage1=%s
' "$CHECKPOINT" "$STAGE1"
    env | LC_ALL=C sort | grep -E '^(ALLOC_|CUDA_|MPS_GPU_ROOT=|TOKENHSI_GPU=)'
    exit 0
fi
# Uses the caller's Python environment; activate tokenhsi_juan/tokenhsi118 first.
mkdir -p "$OUT"
python "$ROOT/TokenHSI-coord/carry_box_swap/configure.py" \
 "$ROOT/TokenHSI-masteer/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" \
 "$OUT/env.yaml" --envs "$ENVS" --skill loco_carry --steps "$EPISODE"
export ALLOC_OUTPUT="$OUT" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID"
export CARRY_PLANNER_PHYSICAL_GPU="$GPU" CARRY_BOX_SWAP_GOALS_FOLLOW_BOX=1
export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0 COORD_VIEWER=0
export CARRY_PLANNER_CONVERGE_PROB=0 CARRY_PLANNER_RANDOMIZE_AGENT_SLOTS=0
export MA_TOKEN=mask MA_TOKENIZER_ZERO=1 MS_MRAND=0 MS_M_LO=0 MS_CLIP=1 MS_ZERO=0
export MS_SCEN=cross MS_SEED="$SEED" MS_REWARD_OUTER=1 MS_POS_C=0.6 MS_VEL_W=1 MS_VIEW_TIMED_CROSS=0
export MS_DRAW_TASK=1 COORD_PRESERVE_PICKUP_APPROACH=0 COORD_ALLOW_HAND_CONTACT=1 STACK_CURRICULUM=0
export MA_SEP=0 MA_SPAWN_GAP=0 MA_C=0 MA_BETA=0 MA_TEAM=share MA_MKSPN=0 MA_TAU=.3
unset MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_S MA_LAYOUT_D MS_SCEN_CURVE MA_METRICS MA_DHIST MS_METRICS
source "$ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"
CHECKPOINT=$(realpath -- "$CHECKPOINT")
STAGE1=$(realpath -- "$STAGE1")
{
 printf 'checkpoint=%q\nstage1=%q\n' "$CHECKPOINT" "$STAGE1"
 env | LC_ALL=C sort | grep -E '^(ALLOC_|CARRY_BOX_SWAP_|CARRY_PLANNER_|COORD_|MA_|MS_|STACK_|CUDA_|MPS_GPU_ROOT=|TOKENHSI_GPU=)'
} > "$OUT/run.env"
cd "$ROOT/TokenHSI-coord"
VIEW_ARGS=(--headless --graphics_device_id -1)
if [ "${ALLOC_VIEW:-0}" = 1 ]; then VIEW_ARGS=(--graphics_device_id "$GPU"); fi
python -u -m task_allocation.train --test "${VIEW_ARGS[@]}" --task HumanoidTaskAllocationMS18 \
 --sim_device cuda:0 --rl_device cuda:0 --physx --pipeline gpu \
 --cfg_train "$ROOT/TokenHSI-masteer/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
 --cfg_env "$OUT/env.yaml" --motion_file "$ROOT/TokenHSI-masteer/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
 --hrl_checkpoint "$STAGE1" --checkpoint "$CHECKPOINT" --num_envs "$ENVS" --seed "$SEED" \
 --output_path "$OUT/unused" > "$OUT/run.log" 2>&1
