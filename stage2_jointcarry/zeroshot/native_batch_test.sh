#!/usr/bin/env bash
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENHSI_GPU=6
export TOKENHSI_CONDA_ENV=tokenhsi
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
source "$ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
set -euo pipefail
# Bypass MPS before the first CUDA initialization; leave the training daemon unchanged.
export CUDA_MPS_PIPE_DIRECTORY=''
unset CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
export JOINT_ZERO_SHOT_GPU_PHYSX=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
# Policy and PhysX both run on GPU 6 in a verified direct CUDA context.
SIM_DEVICE=${NATIVE_PHYSICS_DEVICE:-cuda:0}
PIPELINE=gpu
DEVICE=cuda:0
if [[ "$SIM_DEVICE" == cpu ]]; then
 PIPELINE=cpu
 DEVICE=cpu
 unset JOINT_ZERO_SHOT_GPU_PHYSX
fi
if [[ "$DEVICE" != cpu && "$DEVICE" != cuda:0 ]]; then
    echo 'Policy device must be cpu or cuda:0 (physical GPU 6).' >&2; exit 2
fi
if [[ "$DEVICE" == cpu ]]; then export CUDA_VISIBLE_DEVICES=''; fi
ASSET=${1:?Specify a name from runs/jointcarry_zeroshot/native_assets/catalog.json}
ASSET_ROOT=/home/hwanhee/juan/CVPR2027/runs/jointcarry_zeroshot/native_assets
case "$ASSET" in
 native_source_batch|box|tokenhsi_table_native|tokenhsi_armchair_native|teamhoi_default_round|teamhoi_default_square|teamhoi_default_rectangle|teamhoi_large_round|teamhoi_large_square|teamhoi_large_rectangle) ;;
 *) echo "Unknown native asset: $ASSET" >&2; exit 2 ;;
esac
if [[ "$ASSET" == native_source_batch ]]; then
 export JOINT_ZERO_SHOT_URDF="$ASSET_ROOT/tokenhsi_table_native/asset.urdf"
 export JOINT_ZERO_SHOT_NATIVE_CONFIG="$ASSET_ROOT/source_batch.json"
elif [[ "$ASSET" == box ]]; then
 unset JOINT_ZERO_SHOT_URDF JOINT_ZERO_SHOT_NATIVE_CONFIG
else
 export JOINT_ZERO_SHOT_URDF="$ASSET_ROOT/$ASSET/asset.urdf"
 export JOINT_ZERO_SHOT_NATIVE_CONFIG="$ASSET_ROOT/$ASSET/manifest.json"
fi
export EVAL_SKILLS=loco
OUT=${OUTPUT_PATH:-/home/hwanhee/juan/CVPR2027/runs/jointcarry_zeroshot/${ASSET}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
CKPT=$(python -c 'import json; print(json.load(open("ZEROSHOT_SOURCE.json"))["checkpoint"])')
VIEW_FLAGS=(--headless)
if [[ "${HEADLESS:-1}" == 0 ]]; then VIEW_FLAGS=(--task_camera stack); fi
python tokenhsi/scripts/prepare_joint_carry.py
python tokenhsi/run.py \
 --task HumanoidMACarry \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --checkpoint "$CKPT" --num_agents 2 --num_envs "${NUM_ENVS:-1}" --num_objects 4 \
 --eval_repeats "${EVAL_REPEATS:-3}" --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs 1.0 \
 --task_graph joint_carry_at --episode_length "${EPISODE_LENGTH:-600}" \
 --seed "${SEED:-20261015}" --sim_device "$SIM_DEVICE" --pipeline "$PIPELINE" --rl_device "$DEVICE" \
 "${VIEW_FLAGS[@]}" --no_video --test --eval --output_path "$OUT" 2>&1 | tee "$OUT/eval.log"
