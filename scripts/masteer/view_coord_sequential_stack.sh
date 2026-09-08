#!/bin/bash
# Native Isaac Gym viewer. DISPLAY must point to a desktop or existing X server.
# Usage: COORD_CKPT=/abs/coord_c2_000300.pth bash scripts/masteer/view_coord_sequential_stack.sh /abs/policy.pth [envs]
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
if [ "${1:-}" = --help ]; then
    echo "COORD_CKPT=/abs/coord.pth MA_GPU=0 bash $0 /abs/masteer.pth [envs]"
    echo "Native viewer: DISPLAY required. TOKENHSI_PYTHON overrides the interpreter."
    exit 0
fi
POLICY=${1:?masteer policy.pth required}
ENVS=${ENVS:-${2:-1}}
GPU=${MA_GPU:-0}
STAGE1=${MS_CKPT:-"$REPO/output/ckpt_stage1.pth"}
[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo "envs must be positive" >&2; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "MA_GPU must be 0..7" >&2; exit 2; }
for ckpt in "$POLICY" "$STAGE1"; do
    [ -f "$ckpt" ] || { echo "checkpoint 없음: $ckpt" >&2; exit 2; }
done
POLICY=$(realpath -- "$POLICY")
STAGE1=$(realpath -- "$STAGE1")
. "$ROOT/scripts/masteer/coord_stack_common.sh"
coord_stack_setup
[ -n "${DISPLAY:-}" ] || { echo "viewer를 띄울 DISPLAY가 없습니다. 데스크톱/X 서버에서 실행하세요." >&2; exit 2; }

CFG=$(mktemp /tmp/coord_stack_view.XXXXXX.yaml)
trap 'rm -f -- "$CFG"' EXIT
sed -e 's/^  numAgents:.*/  numAgents: 2/' -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    "$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"
export CUDA_VISIBLE_DEVICES="$GPU"
export MA_TOKEN=${MA_TOKEN:-mask}
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MS_SCEN=${MS_SCEN:-free}
export MS_DBG=0
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_POS_C=${MS_POS_C:-1.2}
export MS_REWARD_OUTER=${MS_REWARD_OUTER:-1}
export MS_VEL_W=${MS_VEL_W:-1}
export STACK_TASK_MODE=stack
export STACK_ALLOW_HAND_CONTACT=${STACK_ALLOW_HAND_CONTACT:-1}
export STACK_FIXED_BOX_SIZE_IDS=${STACK_VIEW_BOX_IDS:-${STACK_FIXED_BOX_SIZE_IDS:-}}
export STACK_BOTTOM_DISPLACE_TOL=${STACK_BOTTOM_DISPLACE_TOL:-0.50}
export STACK_TOP_XY_TOL=${STACK_TOP_XY_TOL:-0.15}
export STACK_TOP_FOLLOWS_BOTTOM=${STACK_TOP_FOLLOWS_BOTTOM:-1}
export STACK_VIRTUAL_RETREAT_BOX=${STACK_VIRTUAL_RETREAT_BOX:-1}
export STACK_EPISODE_LENGTH=${STACK_EPISODE_LENGTH:-900}
export STACK_DEBUG=${STACK_DEBUG:-1}
cd "$REPO"
"$TOKENHSI_PYTHON" -u tokenhsi/run_coord_sequential_stack.py \
    --task HumanoidMACoordSequentialStack \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" --num_envs "$ENVS" \
    --seed "${STACK_EVAL_SEED:-0}" --test --eval_task carry
