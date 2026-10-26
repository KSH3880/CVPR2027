#!/bin/bash
# Isolated upstream snapshot: never load the active PUSH experiment's Python code.
stage1_base_push_run() {
  local mode=$1
  shift
  local scripts root port checkpoint agents envs objects output variant experiment manifest
  scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  root=$(cd "$scripts/../../.." && pwd)
  port="$root/ports/stage1_base_push"
  variant=${TOKENHSI_STAGE1_PUSH_VARIANT:-stage1_base_push}
  case "$variant" in
    stage1_base_push) experiment=Stage1BasePush; manifest=dataset_stage1_base_push.yaml ;;
    stage1_push_finetune) experiment=Stage1PushFinetune; manifest=dataset_stage1_push_finetune.yaml ;;
    *) echo 'Unknown Stage1 PUSH variant' >&2; return 1 ;;
  esac
  export TOKENHSI_GPU=${TOKENHSI_GPU:-1}
  export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_koo}
  if [ "$mode" = train ]; then
    export HEADLESS=1
    unset TOKENHSI_GUI_GPU_MODE TOKENHSI_GUI_GPU_INDEX TOKENHSI_VIEWER_LOW_VRAM
    agents=${1:-2}; envs=${2:-2048}; objects=${3:-4}
    [ "$agents" = 2 ] && [ "$envs" = 2048 ] && [ "$objects" = 4 ] || {
      echo 'Training requires 2 people, 2048 environments and 4 boxes' >&2; return 1;
    }
    local gpu_uuid clients
    gpu_uuid=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=uuid --format=csv,noheader) || return 1
    clients=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader) || return 1
    if printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '
      {gsub(/^[ \t]+|[ \t]+$/, "", $1)}
      $1==gpu && $2~/python/ {found=1} END {exit !found}'; then
      echo 'Selected GPU already has a Python CUDA client; no process was stopped.' >&2; return 1
    fi
    output=${OUTPUT_PATH:-output/stage1_base_push}
    set -- --headless --task_graph random_scenario
    if [ -n "${TOKENHSI_TRAIN_EPISODE_LENGTH:-}" ]; then
      case "$TOKENHSI_TRAIN_EPISODE_LENGTH" in
        *[!0-9]*|0*) echo 'TOKENHSI_TRAIN_EPISODE_LENGTH must be a positive integer without leading zeros' >&2; return 1 ;;
      esac
      set -- "$@" --episode_length "$TOKENHSI_TRAIN_EPISODE_LENGTH"
    fi
    if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
    if [ -n "${RESUME_CHECKPOINT:-}" ]; then
      checkpoint=$(realpath -e "$RESUME_CHECKPOINT") || return 1
      set -- "$@" --checkpoint "$checkpoint" --resume 1
    fi
  else
    [ $# -ge 1 ] && [ -f "$1" ] || { echo 'Provide a Stage1BasePush checkpoint' >&2; return 1; }
    checkpoint=$(realpath -e "$1") || return 1
    agents=${2:-2}; envs=${3:-1}; objects=${4:-4}
    [ "$agents" = 2 ] && [ "$objects" = 4 ] || { echo 'Evaluation requires 2 people and 4 boxes' >&2; return 1; }
    local repeats=${5:-10}
    output=${OUTPUT_PATH:-output/stage1_base_push_eval}
    if [ "${HEADLESS:-0}" = 0 ]; then . "$scripts/gui_gpu_env.sh"; fi
    set -- --checkpoint "$checkpoint" --test --eval --eval_repeats "$repeats" \
      --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
      --task_graph "${TASK_GRAPH:-push_at}" --task_camera stack
    if [ "${HEADLESS:-0}" != 0 ]; then set -- "$@" --headless; fi
    if [ -n "${TOKENHSI_VIEWER_EPISODE_LENGTH:-}" ]; then
      case "$TOKENHSI_VIEWER_EPISODE_LENGTH" in
        *[!0-9]*|0) echo 'TOKENHSI_VIEWER_EPISODE_LENGTH must be a positive integer' >&2; return 1 ;;
      esac
      set -- "$@" --episode_length "$TOKENHSI_VIEWER_EPISODE_LENGTH"
    fi
  fi
  . "$scripts/runtime_env.sh"
  case "$output" in /*) ;; *) output="$root/$output" ;; esac
  output=$(realpath -m "$output")
  case "$output" in "$root"/output/stage1_base_push*) ;; *)
    echo 'Use a separate output/stage1_base_push* output directory' >&2; return 1 ;;
  esac
  export PYTHONPATH="$port/tokenhsi:$port${PYTHONPATH:+:$PYTHONPATH}"
  cd "$port"
  if [ -n "${checkpoint:-}" ]; then python preflight.py "$checkpoint" "$mode" "$variant" || return 1; fi
  if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
  python tokenhsi/run.py --task HumanoidMAStage1Push \
    --cfg_env "tokenhsi/data/cfg/multi_agent/$variant.yaml" \
    --cfg_train "tokenhsi/data/cfg/train/rlg/$variant.yaml" \
    --motion_file "tokenhsi/data/$manifest" \
    --num_agents "$agents" --num_envs "$envs" --num_objects "$objects" \
    --experiment "$experiment" --output_path "$output" --no_video "$@"
}
