#!/bin/bash
# Isolated upstream snapshot: never load the active PUSH experiment's Python code.
stage1_base_push_run() {
  local mode=$1
  shift
  local scripts root port checkpoint agents envs objects output
  scripts="$KOO_PORT/tokenhsi/scripts/multi_agent"
  root=$(cd "$scripts/../../.." && pwd)
  port="$root/ports/stage1_base_push"
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
      if [ "${KOO_ALLOW_SHARED_MPS:-0}" != 1 ]; then
        echo 'Selected GPU already has a Python CUDA client; no process was stopped. Explicit verified sharing requires KOO_ALLOW_SHARED_MPS=1.' >&2; return 1
      fi
      local available existing_count mps_count
      available=$(nvidia-smi --id="$TOKENHSI_GPU" --query-gpu=memory.free --format=csv,noheader,nounits) || return 1
      existing_count=$(printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '{gsub(/^[ \t]+|[ \t]+$/, "", $1)} $1==gpu && $2~/python/ {n++} END {print n+0}')
      mps_count=$(printf '%s\n' "$clients" | awk -F, -v gpu="$gpu_uuid" '{gsub(/^[ \t]+|[ \t]+$/, "", $1)} $1==gpu && $2~/nvidia-cuda-mps-server/ {n++} END {print n+0}')
      [ "$available" -ge 32000 ] && [ "$existing_count" -eq 1 ] && [ "$mps_count" -eq 1 ] || {
        echo 'Shared training requires >=32GiB free, one Python client and an existing MPS server.' >&2; return 1;
      }
      echo "[shared GPU] opt-in GPU=$TOKENHSI_GPU free=${available}MiB; existing client/server unchanged"

    fi
    output=${OUTPUT_PATH:-output/stage1_base_push}
    set -- --headless --task_graph random_scenario
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
  fi
  . "$scripts/runtime_env.sh"
  if [ "$mode" = train ] && [ "${KOO_ALLOW_SHARED_MPS:-0}" = 1 ]; then
    [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] || { echo 'Verified MPS attachment missing; refusing shared run.' >&2; return 1; }
  fi
  case "$output" in /*) ;; *) output="$root/$output" ;; esac
  output=$(realpath -m "$output")
  case "$output" in "$KOO_PROJECT"/runs/jointpush/koo_1021/output/stage1_base_push*) ;; *)
    echo 'Use a separate output/stage1_base_push* output directory' >&2; return 1 ;;
  esac
  export PYTHONPATH="$port/tokenhsi:$port${PYTHONPATH:+:$PYTHONPATH}"
  cd "$port"
  if [ -n "${checkpoint:-}" ]; then python preflight.py "$checkpoint" "$mode" || return 1; fi
  if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
  python tokenhsi/run.py --task HumanoidMAStage1Push \
    --cfg_env tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml \
    --cfg_train tokenhsi/data/cfg/train/rlg/stage1_base_push.yaml \
    --motion_file tokenhsi/data/dataset_stage1_base_push.yaml \
    --num_agents "$agents" --num_envs "$envs" --num_objects "$objects" \
    --experiment Stage1BasePush --output_path "$output" --no_video "$@"
}

stage1_base_push_run "$@"
