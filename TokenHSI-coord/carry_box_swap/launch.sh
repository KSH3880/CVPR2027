#!/bin/bash
# Usage: launch.sh new-tag loco_carry|carryWith [original-stage1.pth]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: launch.sh new-tag skill [stage1.pth]}
SKILL=${2:?usage: launch.sh new-tag skill [stage1.pth]}
CHECKPOINT=${3:-"$ROOT/TokenHSI-masteer/output/ckpt_stage1.pth"}
GPU=${MA_GPU:-0}
ENVS=${CARRY_BOX_SWAP_ENVS:-16}
STEPS=${CARRY_BOX_SWAP_STEPS:-600}
SEED=${CARRY_BOX_SWAP_SEED:-0}
case "$TAG" in ""|*[!A-Za-z0-9_.-]*) exit 2;; esac
case "$SKILL" in loco_carry|carryWith) ;; *) echo "unknown skill" >&2; exit 2;; esac
[ -f "$CHECKPOINT" ] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
OUT="$ROOT/runs/carry_box_swap/$TAG"
[ ! -e "$OUT" ] || { echo "use a new tag: $OUT" >&2; exit 3; }
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
set +u
source "${CONDA_BASE:-/home/injesus1010/anaconda3}/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"
set -u
CHECKPOINT=$(realpath -- "$CHECKPOINT")
mkdir -p "$OUT"
python "$ROOT/TokenHSI-coord/carry_box_swap/configure.py" \
 "$ROOT/TokenHSI-coord/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" \
 "$OUT/env.yaml" --envs "$ENVS" --skill "$SKILL" --steps "$STEPS"
export CARRY_BOX_SWAP_OUTPUT="$OUT" CARRY_BOX_SWAP_SKILL="$SKILL" CARRY_BOX_SWAP_STEPS="$STEPS"
export CARRY_BOX_SWAP_GOALS_FOLLOW_BOX=${CARRY_BOX_SWAP_GOALS_FOLLOW_BOX:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID"
export MA_LAYOUT=far MA_LAYOUT_L=6 MA_LAYOUT_S=2.5 MA_SEP=0 MA_SPAWN_GAP=1
export MA_C=0 MA_BETA=0 MA_TEAM=share MA_MKSPN=0 MA_TAU=.3
unset MA_METRICS MA_DHIST MS_METRICS
source "$ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"
{
 printf 'checkpoint=%q\n' "$CHECKPOINT"
 env | LC_ALL=C sort | grep -E '^(CARRY_BOX_SWAP_|MA_|CUDA_VISIBLE_DEVICES=)'
} > "$OUT/run.env"
cd "$ROOT/TokenHSI-coord"
VIEW_ARGS=(--headless --graphics_device_id -1)
if [ "${CARRY_BOX_SWAP_VIEW:-0}" = 1 ]; then VIEW_ARGS=(--graphics_device_id "$GPU"); fi
python -u -m carry_box_swap.run --test "${VIEW_ARGS[@]}" --task HumanoidCarryBoxSwapProbe \
 --sim_device cuda:0 --rl_device cuda:0 --physx --pipeline gpu \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
 --cfg_env "$OUT/env.yaml" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --checkpoint "$CHECKPOINT" --num_envs "$ENVS" --seed "$SEED" \
 --output_path "$OUT/unused" > "$OUT/run.log" 2>&1
