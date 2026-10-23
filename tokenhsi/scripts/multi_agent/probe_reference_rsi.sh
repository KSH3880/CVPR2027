#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in 6|7) ;; *) echo "Only GPUs 6 and 7 are allowed" >&2; exit 2;; esac
RSI_PHYSICAL_GPU=$TOKENHSI_GPU
export CUDA_VISIBLE_DEVICES=$TOKENHSI_GPU
. "$(dirname "$0")/runtime_env.sh"
export RSI_PROBE_MODE=${1:-baseline}
case "$RSI_PROBE_MODE" in
 baseline|raw_time_late|impact_sweep) CFG=tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_random_binding.yaml;;
 time_late|semantic_late) CFG=tokenhsi/data/cfg/multi_agent/approach_reference_rsi_${RSI_PROBE_MODE}.yaml;;
 *) echo "Use baseline, raw_time_late, time_late, semantic_late or impact_sweep" >&2; exit 2;;
esac
CFG=${RSI_PROBE_CFG:-$CFG}
export RSI_PROBE_OUTPUT=${RSI_PROBE_OUTPUT:-../runs/reference_rsi/probe_${RSI_PROBE_MODE}}
python tokenhsi/scripts/multi_agent/probe_reference_rsi.py --task HumanoidMACarry \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation.yaml \
 --cfg_env "$CFG" --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --num_envs 2048 --num_agents 2 --num_objects 4 --seed "${SEED:-0}" \
 --output_path "$RSI_PROBE_OUTPUT/runtime" --experiment ReferenceRSIProbe \
 --task_graph "${TASK_GRAPH:-random_scenario}" --headless --no_video --max_iterations 1
