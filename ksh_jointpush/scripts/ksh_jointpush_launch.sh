#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SOURCE="$ROOT/ksh_jointpush"
WORK="$ROOT/runs/jointpush/ksh_jointpush_rsi/workspace_push_solo45_coupled0"
MODE=${1:-train}
export KSH_RSI_MODE=${KSH_RSI_MODE:-group}
case "$KSH_RSI_MODE" in
 group) ;;
 agent) WORK="${WORK}_agent_rsi" ;;
 balanced) WORK="${WORK}_balanced_rsi" ;;
 *) echo 'KSH_RSI_MODE must be group, agent or balanced'; exit 2 ;;
esac
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
gpu_index=$(nvidia-smi -i "$TOKENHSI_GPU" --query-gpu=index --format=csv,noheader,nounits)
[ "$gpu_index" = 6 ] || { echo 'This prepared launcher is restricted to GPU6'; exit 2; }
export TOKENHSI_CONDA_ENV=tokenhsi_juan PYTHONDONTWRITEBYTECODE=1
PYTHON="$HOME/anaconda3/envs/tokenhsi_juan/bin/python"
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/tokenhsi_juan/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/ksh_jointpush/scripts/ksh_jointpush_prepare.py" --rsi-mode "$KSH_RSI_MODE"
export HEADLESS=${HEADLESS:-1}
if [ "$MODE" = train ]; then export HEADLESS=1; fi
if [ "$MODE" = check ]; then
 export HEADLESS=1 KSH_JOINTPUSH_CHECK=1 MPS_GPU_ROOT=/tmp/ksh-jointpush-cpu-no-mps
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
elif [ "$HEADLESS" = 0 ]; then
 export MPS_GPU_ROOT=/tmp/ksh-jointpush-native-no-mps
 unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
. "$SOURCE/tokenhsi/scripts/multi_agent/runtime_env.sh"
export TORCH_EXTENSIONS_DIR="$ROOT/runs/jointpush/torch_extensions" CXX=/usr/bin/g++-11
export PYTHONPATH="$SOURCE/tokenhsi:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="$ROOT/jointpush/checkpoints/ksh_jointpush/Stage1PushFinetune_00014000.pth"
args=(--task HumanoidMAStage2JointPushRsi --cfg_env "$WORK/configs/env.yaml" --cfg_train "$WORK/configs/train.yaml" --motion_file "$WORK/configs/motions.yaml" --num_agents 4 --num_objects 8 --seed "${SEED:-42}" --no_video)
if [ "$MODE" = train ]; then
 [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] || { echo 'Verified per-GPU MPS attachment required'; exit 1; }
 args+=(--headless --sim_device cuda:0 --rl_device cuda:0 --num_envs 1024 --task_graph random_scenario)
 if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--checkpoint "$RESUME_CHECKPOINT" --resume 1); else args+=(--transfer_checkpoint "$CHECKPOINT"); fi
 OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/ksh_jointpush/output/train_${KSH_RSI_MODE}_$(date +%Y%m%d_%H%M%S)}
else
 args+=(--test --eval --checkpoint "${CHECKPOINT_OVERRIDE:-$CHECKPOINT}" --num_envs "${NUM_ENVS:-1}" --task_graph "${TASK_GRAPH:-random_scenario}" --eval_repeats "${REPEATS:-100000}")
 case "${PUSH_RSI:-0}" in
  0) ;;
  1) args+=(--eval_push_rsi) ;;
  *) echo 'PUSH_RSI must be 0 or 1'; exit 2 ;;
 esac
 OUTPUT_PATH=${OUTPUT_PATH:-$ROOT/ksh_jointpush/output_etc/check/view_$(date +%Y%m%d_%H%M%S)}
 if [ "$MODE" = check ]; then
  export CUDA_VISIBLE_DEVICES=''
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  args+=(--headless --sim_device cpu --pipeline cpu --rl_device cpu --eval_push_rsi)
 elif [ "$HEADLESS" = 0 ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  args+=(--sim_device "cuda:$TOKENHSI_GUI_GPU_INDEX" --rl_device "cuda:$TOKENHSI_GUI_GPU_INDEX")
 else
  args+=(--headless --sim_device cuda:0 --rl_device cuda:0)
 fi
fi
cd "$SOURCE"
exec "$PYTHON" -u "$ROOT/ksh_jointpush/scripts/ksh_jointpush_run.py" "${args[@]}" --output_path "$OUTPUT_PATH"
