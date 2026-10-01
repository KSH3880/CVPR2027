#!/bin/bash
# Usage: launch.sh new-tag loco_carry|carryWith [ms18-policy.pth]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: launch.sh new-tag skill [ms18-policy.pth]}
SKILL=${2:?usage: launch.sh new-tag skill [ms18-policy.pth]}
CHECKPOINT=${3:-"$ROOT/TokenHSI-masteer/output/ms18_maskteam_origscale_c06_s0_00009000.pth"}
STAGE1=${MS_CKPT:-"$ROOT/TokenHSI-masteer/output/ckpt_stage1.pth"}
[ -f "$STAGE1" ] || { echo "missing stage1: $STAGE1" >&2; exit 2; }
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
export CARRY_BOX_SWAP_EXECUTOR=ms18 CARRY_PLANNER_PHYSICAL_GPU="$GPU"
export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0 COORD_VIEWER=0
export CARRY_PLANNER_CONVERGE_PROB=0 CARRY_PLANNER_RANDOMIZE_AGENT_SLOTS=0
export MA_TOKEN=mask MA_TOKENIZER_ZERO=1
export MS_MRAND=0 MS_M_LO=0 MS_CLIP=1 MS_ZERO=0 MS_SCEN=cross MS_SEED="$SEED"
export MS_REWARD_OUTER=1 MS_POS_C=0.6 MS_VEL_W=1 MS_VIEW_TIMED_CROSS=0
export MS_DRAW_TASK=1
export COORD_PRESERVE_PICKUP_APPROACH=0 COORD_ALLOW_HAND_CONTACT=1
unset MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_S MA_LAYOUT_D MS_SCEN_CURVE
export MA_SEP=0 MA_SPAWN_GAP=1
export MA_C=0 MA_BETA=0 MA_TEAM=share MA_MKSPN=0 MA_TAU=.3
unset MA_METRICS MA_DHIST MS_METRICS
source "$ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"
{
 printf 'checkpoint=%q\nstage1=%q\n' "$CHECKPOINT" "$STAGE1"
 env | LC_ALL=C sort | grep -E '^(CARRY_BOX_SWAP_|CARRY_PLANNER_|COORD_|MA_|MS_|CUDA_VISIBLE_DEVICES=)'
} > "$OUT/run.env"
cd "$ROOT/TokenHSI-coord"
VIEW_ARGS=(--headless --graphics_device_id -1)
if [ "${CARRY_BOX_SWAP_VIEW:-0}" = 1 ]; then VIEW_ARGS=(--graphics_device_id "$GPU"); fi
python -u -m carry_box_swap.run --test "${VIEW_ARGS[@]}" --task HumanoidMACarryBoxSwapMS18 \
 --sim_device cuda:0 --rl_device cuda:0 --physx --pipeline gpu \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
 --cfg_env "$OUT/env.yaml" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --hrl_checkpoint "$STAGE1" --checkpoint "$CHECKPOINT" --num_envs "$ENVS" --seed "$SEED" \
 --output_path "$OUT/unused" > "$OUT/run.log" 2>&1
