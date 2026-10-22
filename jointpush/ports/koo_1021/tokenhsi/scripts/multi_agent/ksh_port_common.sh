#!/bin/bash
# Private ksh snapshot; never import the main checkout's env/learning/utils.
ksh_port_run() {
  local mode=$1 variant=$2
  shift 2
  export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
  . "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
  set -eu
  local port_root="$TOKENHSI_ROOT/ports/ksh_stage1"
  local stem="approach_scenario_stage1_unified_${variant}"
  local train_cfg="amp_ma_carry_relation_unified_${variant}.yaml"
  local experiment
  case "$variant" in
    size_rsi) train_cfg=amp_ma_carry_relation.yaml; experiment=ApproachScenarioStage1UnifiedSizeRsi ;;
    size_rsi_typed_bias) experiment=ApproachScenarioStage1UnifiedSizeRsiTypedBias ;;
    size_rsi_typed_bias_message) experiment=ApproachScenarioStage1UnifiedSizeRsiTypedBiasMessage ;;
    *) echo "Unknown ksh port variant: $variant" >&2; return 1 ;;
  esac
  local output="${OUTPUT_PATH:-output/ksh_${variant}}"
  case "$output" in /*) ;; *) output="$TOKENHSI_ROOT/$output" ;; esac
  output=$(realpath -m "$output")
  # Protect all pre-existing main/ksh experiment outputs from accidental reuse.
  case "$output" in "$TOKENHSI_ROOT"/output/ksh_*) ;; *)
    echo "Ksh port OUTPUT_PATH must be inside $TOKENHSI_ROOT/output/ksh_*" >&2
    return 1 ;;
  esac
  export OUTPUT_PATH="$output"
  export PYTHONPATH="$port_root/tokenhsi:$port_root:$port_root/tokenhsi/tests${PYTHONPATH:+:$PYTHONPATH}"
  local agents envs objects checkpoint
  if [ "$mode" = train ]; then
    agents=${1:-2}; envs=${2:-2048}; objects=${3:-4}
    set -- --task_graph random_scenario --seed "${SEED:-42}" --headless
    if [ -n "${MAX_ITERATIONS:-}" ]; then set -- "$@" --max_iterations "$MAX_ITERATIONS"; fi
    if [ -n "${RESUME_CHECKPOINT:-}" ]; then
      checkpoint=$(realpath -e "$RESUME_CHECKPOINT")
      set -- "$@" --checkpoint "$checkpoint" --resume 1
    fi
  else
    if [ $# -lt 1 ] || [ ! -f "$1" ]; then
      echo "usage: <checkpoint.pth> [agents=2] [envs=16] [objects=4] [repeats=3]" >&2
      return 1
    fi
    checkpoint=$(realpath -e "$1")
    agents=${2:-2}; envs=${3:-16}; objects=${4:-4}
    local repeats=${5:-3}
    set -- --checkpoint "$checkpoint" --test --eval --eval_repeats "$repeats" \
      --eval_skills "${EVAL_SKILLS:-loco}" --eval_skill_probs "${EVAL_SKILL_PROBS:-1.0}" \
      --task_graph "${TASK_GRAPH:-random_scenario}" --task_camera "${TASK_CAMERA:-stack}"
    if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
    if [ "${TASK_ROLE_SWAP:-0}" = 1 ]; then set -- "$@" --task_role_swap; fi
    if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
    if [ -n "${SEED:-}" ]; then set -- "$@" --seed "$SEED"; fi
  fi
  cd "$port_root"
  python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train "tokenhsi/data/cfg/train/rlg/$train_cfg" \
    --cfg_env "tokenhsi/data/cfg/multi_agent/$stem.yaml" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_agents "$agents" --num_envs "$envs" --num_objects "$objects" \
    --output_path "$output" --experiment "$experiment" --no_video "$@"
}

ksh_port_vnc() {
  local variant=$1
  shift
  set -eu
  local script_dir root checkpoint
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  root=$(cd "$script_dir/../../.." && pwd)
  cd "$root"
  if [ $# -lt 1 ] || [ $# -gt 5 ] || [ ! -f "$1" ]; then
    echo "Provide an existing checkpoint.pth relative to the repository root." >&2
    return 1
  fi
  checkpoint=$(realpath -e "$1")
  export HEADLESS=0
  export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi}
  export VNC_DIR=${VNC_DIR:-$HOME/opt/vnc}
  exec bash "$script_dir/run-gui.sh" \
    bash "$script_dir/ksh_${variant}_test.sh" \
    "$checkpoint" "${2:-2}" "${3:-1}" "${4:-4}" "${5:-10}"
}
