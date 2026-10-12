#!/bin/bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
case "$TOKENHSI_GPU" in 6|7) ;; *) echo 'Only GPUs 6 and 7 allowed' >&2; exit 2;; esac
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash probe_student_candidates.sh /absolute/student.pth /absolute/candidates.json' >&2
  exit 2
fi
export RSI_STUDENT_CHECKPOINT=$(realpath "$1")
export RSI_CANDIDATES=$(realpath "$2")
test -f "$RSI_STUDENT_CHECKPOINT"
test -f "$RSI_CANDIDATES"
export CUDA_VISIBLE_DEVICES=$TOKENHSI_GPU
. "$(dirname "$0")/runtime_env.sh"
export RSI_PROBE_MODE=candidate_replay
export RSI_PROBE_ROUNDS=${RSI_PROBE_ROUNDS:-32}
export RSI_PROBE_STEPS=${RSI_PROBE_STEPS:-20}
export RSI_PROBE_OUTPUT=${RSI_PROBE_OUTPUT:-../runs/reference_rsi/student_candidates_seed1}
if [ -e "$RSI_PROBE_OUTPUT/summary.json" ]; then
  echo 'Output already completed; choose a new RSI_PROBE_OUTPUT' >&2
  exit 2
fi
mkdir -p "$RSI_PROBE_OUTPUT"
# Numbered checkpoint is frozen for the entire run and retained as provenance.
if [ -e "$RSI_PROBE_OUTPUT/student_snapshot.pth" ]; then
  echo 'Snapshot exists; choose a new output directory' >&2
  exit 2
fi
cp "$RSI_STUDENT_CHECKPOINT" "$RSI_PROBE_OUTPUT/student_snapshot.pth"
export RSI_STUDENT_CHECKPOINT=$(realpath "$RSI_PROBE_OUTPUT/student_snapshot.pth")
python tokenhsi/scripts/multi_agent/probe_reference_rsi.py --task HumanoidMACarry \
 --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_shared_edge_encoder.yaml \
 --cfg_env "${RSI_STUDENT_CFG:-../runs/manual_resume_289953/shared_random_binding_full_amp.yaml}" \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --num_envs "${RSI_PROBE_ENVS:-2048}" --num_agents 2 --num_objects 4 --seed "${SEED:-1}" \
 --output_path "$RSI_PROBE_OUTPUT/runtime" --experiment StudentCandidateProbe \
 --task_graph random_scenario --headless --no_video --max_iterations 1 --minibatch_size 4096
