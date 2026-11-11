#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime="${TOKENHSI_RUNTIME:-/home/injesus1010/anaconda3/envs/tokenhsi118}"
export PATH="$runtime/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD:$PWD/tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CXX=/usr/bin/g++-11 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR="$PWD/output_etc/check/torch_extensions"
export CUDA_VISIBLE_DEVICES="${TOKENHSI_GPU:-0}"
export PUSH_EVAL_OUT="${PUSH_EVAL_OUT:-$PWD/output_etc/check/submission_inference/push_eval.json}"
args=()
if [[ "${HEADLESS:-1}" == 1 ]]; then args+=(--headless); fi
if [[ "${HEADLESS:-1}" == 0 ]]; then
  # Match the working local Isaac Gym viewer: select NVIDIA's Vulkan ICD.
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
  export DRI_PRIME="${TOKENHSI_GPU:-0}!"
  export VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select
  export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
  unset NODEVICE_SELECT MESA_VK_DEVICE_SELECT
fi
export VIEW_LOOP="${VIEW_LOOP:-$((1 - ${HEADLESS:-1}))}"
exec "$runtime/bin/python" scripts/infer_submission.py \
 --task HumanoidTokenHSIPushKD --cfg_env tokenhsi/data/cfg/multi_task/push_kd.yaml \
 --cfg_train tokenhsi/data/cfg/train/rlg/push_kd.yaml --motion_file tokenhsi/data/dataset_push_kd.yaml \
 --test --eval --eval_task push \
 --checkpoint "${CHECKPOINT:-$PWD/push_kd_submission_20261105/model/TokenHSIPushKD_B_seed42_e11600.pth}" \
 --num_envs "${NUM_ENVS:-4}" "${args[@]}" --sim_device "${SIM_DEVICE:-cuda:0}" --rl_device "${RL_DEVICE:-cuda:0}" --pipeline "${SIM_PIPELINE:-gpu}" --seed "${SEED:-0}"
