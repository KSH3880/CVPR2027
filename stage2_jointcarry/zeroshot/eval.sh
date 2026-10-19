#!/usr/bin/env bash
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENHSI_GPU=6
export TOKENHSI_CONDA_ENV=tokenhsi
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
source "$ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
set -euo pipefail
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
# New mesh physics stays on CPU while GPU 6 remains shared by MPS learners.
DEVICE=${ZERO_SHOT_POLICY_DEVICE:-cpu}
if [[ "$DEVICE" != cpu && "$DEVICE" != cuda:0 ]]; then
    echo 'Policy device must be cpu or cuda:0 (physical GPU 6).' >&2; exit 2
fi
if [[ "$DEVICE" == cpu ]]; then export CUDA_VISIBLE_DEVICES=''; fi
ASSET=${1:-box}
if [[ "$ASSET" == box ]]; then
    unset JOINT_ZERO_SHOT_URDF
elif [[ "$ASSET" == round_table ]]; then
    export JOINT_ZERO_SHOT_URDF="$ROOT/zeroshot/assets/round_table/asset.urdf"
else
    echo 'Expected box or round_table' >&2; exit 2
fi
OUT=${OUTPUT_PATH:-/home/hwanhee/juan/CVPR2027/runs/jointcarry_zeroshot/${ASSET}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
CKPT=$(python -c 'import json; print(json.load(open("ZEROSHOT_SOURCE.json"))["checkpoint"])')
python tokenhsi/scripts/prepare_joint_carry.py
python tokenhsi/run.py \
 --task HumanoidMACarry \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --checkpoint "$CKPT" --num_agents 2 --num_envs "${NUM_ENVS:-1}" --num_objects 4 \
 --eval_repeats "${EVAL_REPEATS:-3}" --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs 1.0 \
 --task_graph joint_carry_at --episode_length "${EPISODE_LENGTH:-600}" \
 --seed "${SEED:-20261015}" --sim_device cpu --pipeline cpu --rl_device "$DEVICE" \
 --headless --no_video --test --eval --output_path "$OUT" 2>&1 | tee "$OUT/eval.log"
