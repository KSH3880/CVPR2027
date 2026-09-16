#!/bin/bash
# Planner-free fixed-path test. Usage: view_midpath_release.sh frozen-agent.pth [envs]
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
COORD="$ROOT/TokenHSI-coord"
EXEC_REPO="$ROOT/TokenHSI-masteer"
POLICY=${1:?usage: view_midpath_release.sh <frozen-agent.pth> [envs]}
ENVS=${ENVS:-${2:-1}}
# This local diagnostic defaults to physical GPU 1. It deliberately does not
# remap CUDA ordinals, so every cuda:$GPU below names that physical GPU.
GPU=${MA_GPU:-1}
STAGE1=${MS_CKPT:-"$EXEC_REPO/output/ckpt_stage1.pth"}
[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "envs must be positive" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "MA_GPU must be non-negative" >&2; exit 2; }
for file in "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint 없음: $file" >&2; exit 2; }
done
POLICY=$(realpath -- "$POLICY")
STAGE1=$(realpath -- "$STAGE1")
# Do not CUDA-remap this GUI diagnostic: Isaac Gym's graphics ordinal and
# PyTorch/PhysX ordinals must name the same physical card. The Python entry
# also calls torch.cuda.set_device(STACK_VIEW_GPU) before TokenHSI is loaded.
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID STACK_VIEW_GPU="$GPU"
export TOKENHSI_GRAPHICS_DEVICE_ID="$GPU"
export DISPLAY=${DISPLAY:-:0}
if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_BASE=$(conda info --base)
    else CONDA_BASE=/home/injesus1010/anaconda3
    fi
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi118}"
CFG=$(mktemp /tmp/midpath_release_view.XXXXXX.yaml)
cleanup() { rm -f -- "$CFG"; }
trap cleanup EXIT INT TERM
sed -e 's/^  numAgents:.*/  numAgents: 2/' \
    -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    -e 's/^  envSpacing:.*/  envSpacing: 5/' \
    -e 's/^  enableDebugVis:.*/  enableDebugVis: True/' \
    "$EXEC_REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"
export MA_TOKEN=mask MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MS_MRAND=${MS_MRAND:-4} MS_M_LO=${MS_M_LO:-0.25}
export MS_CLIP=1 MS_ZERO=0 MS_SCEN=${MS_SCEN:-free} MS_DBG=${MS_DBG:-0}
export MS_REWARD_OUTER=1 MS_POS_C=${MS_POS_C:-1.2} MS_VEL_W=1
# The box->endpoint leg must pass through the unchanged carry goal exactly.
# Curvature would only encourage proximity and invalidate this diagnostic.
export MS_LAT_MAX=0
export STACK_MIDPATH_EXTENSION=${STACK_MIDPATH_EXTENSION:-2.0}
export STACK_CUDA_SYNC_DEBUG=${STACK_CUDA_SYNC_DEBUG:-1}
. "$COORD/stack_planner/physx_cuda_compat.sh"
echo "planner: disabled"
echo "agent: $POLICY"
echo "compute/rl: physical cuda:$GPU (no CUDA_VISIBLE_DEVICES remap)"
echo "graphics: physical device $TOKENHSI_GRAPHICS_DEVICE_ID"
echo "task: plain HumanoidMASteerCarry (no stack coordinator/phases)"
echo "path: root -> box -> carry goal -> ${STACK_MIDPATH_EXTENSION}m beyond goal"
cd "$COORD"
python -u -m stack_planner.run_midpath_release \
    --task HumanoidMAMidpathReleaseView \
    --sim_device "cuda:$GPU" --rl_device "cuda:$GPU" \
    --graphics_device_id "$TOKENHSI_GRAPHICS_DEVICE_ID" --physx --pipeline gpu \
    --cfg_train "$EXEC_REPO/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
    --cfg_env "$CFG" \
    --motion_file "$EXEC_REPO/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" \
    --num_envs "$ENVS" --seed "${STACK_EVAL_SEED:-0}" --test --eval_task carry
