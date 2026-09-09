#!/bin/bash
# Simultaneous two-agent C13 coordinator viewer. No sequential-stack phases.
# Usage: MA_GPU=0 bash scripts/masteer/view_coord_carry.sh /abs/Humanoid_*.pth [envs]
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REPO="$ROOT/TokenHSI-masteer"
POLICY=${1:?usage: view_coord_carry.sh <masteer-policy.pth> [envs]}
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
[ -n "${DISPLAY:-}" ] || { echo "viewer를 띄울 DISPLAY가 없습니다." >&2; exit 2; }

CFG=$(mktemp /tmp/coord_carry_view.XXXXXX.yaml)
trap 'rm -f -- "$CFG"' EXIT
sed -e 's/^  numAgents:.*/  numAgents: 2/' -e "s/^  numEnvs:.*/  numEnvs: $ENVS/" \
    "$REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" > "$CFG"

export CUDA_VISIBLE_DEVICES="$GPU"
export COORD_PROVIDER=learned
export COORD_MODEL=c2
export COORD_VIEWER=1
export COORD_DRAW_CANDIDATES=${COORD_DRAW_CANDIDATES:-1}
export COORD_SPEED_LIMIT=${COORD_SPEED_LIMIT:-1}
export COORD_ACCEL_UP=${COORD_ACCEL_UP:-0.75}
export COORD_ACCEL_DOWN=${COORD_ACCEL_DOWN:-1.0}
export COORD_ALLOW_HAND_CONTACT=${COORD_ALLOW_HAND_CONTACT:-1}
export COORD_PRESERVE_PICKUP_APPROACH=${COORD_PRESERVE_PICKUP_APPROACH:-1}
export COORD_GRASP_STABLE_STEPS=${COORD_GRASP_STABLE_STEPS:-12}
export MA_TOKEN=${MA_TOKEN:-mask}
export MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1}
export MA_SEP=${MA_SEP:-0}
export MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0}
export MS_MRAND=${MS_MRAND:-4}
export MS_M_LO=${MS_M_LO:-0.25}
export MS_ZERO=0
export MS_CLIP=1
export MS_DBG=${MS_DBG:-1}
export MS_DRAW_SPEED=${MS_DRAW_SPEED:-1}
export MS_CAM=${MS_CAM:-top}
export MS_CAM_H=${MS_CAM_H:-17}
export MS_CAM_B=${MS_CAM_B:-9}
# Match the normal training/evaluation distribution by default.  Timed cross
# is an explicit stress scene because it also changes the initial geometry.
export MS_SCEN=${MS_SCEN:-free}
export MS_VIEW_TIMED_CROSS=${MS_VIEW_TIMED_CROSS:-0}
export MS_VIEW_TIMED_CROSS_PROB=${MS_VIEW_TIMED_CROSS_PROB:-1.0}
export MS_VIEW_TIMED_CROSS_TOL=${MS_VIEW_TIMED_CROSS_TOL:-0.25}

echo "=============================================================="
echo " task        HumanoidMACoordCarry (simultaneous; no sequential stack)"
echo " policy      $POLICY (weights only)"
echo " coordinator $COORD_CKPT"
echo " scene       $MS_SCEN timed_cross=$MS_VIEW_TIMED_CROSS"
echo " hand contact allowed=$COORD_ALLOW_HAND_CONTACT"
echo " pickup path native=$COORD_PRESERVE_PICKUP_APPROACH"
echo " grasp stable steps=$COORD_GRASP_STABLE_STEPS"
echo "=============================================================="

cd "$REPO"
"$TOKENHSI_PYTHON" -u tokenhsi/run_coord_carry.py \
    --task HumanoidMACoordCarry \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 --physx --pipeline gpu \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "$STAGE1" --checkpoint "$POLICY" --num_envs "$ENVS" \
    --seed "${MS_SEED:-0}" --test --eval --eval_task carry
