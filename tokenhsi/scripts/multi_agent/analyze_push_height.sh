#!/bin/bash
# Analysis only; no policy/optimizer/checkpoint changes.
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-6}
export TOKENHSI_CONDA_ENV=tokenhsi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
. "$(dirname "$0")/runtime_env.sh"
export PYTHONPATH="tokenhsi${PYTHONPATH:+:$PYTHONPATH}"
ANALYSIS_OUT=${OUTPUT_PATH:-output/push_height_analysis_$(date +%Y%m%d_%H%M%S)}
if [ -e "$ANALYSIS_OUT" ]; then echo 'Use a new OUTPUT_PATH.' >&2; exit 2; fi
mkdir -p "$ANALYSIS_OUT"
python tokenhsi/scripts/multi_agent/analyze_push_height_forces.py --output "$ANALYSIS_OUT/known_forces" > "$ANALYSIS_OUT/known_forces.log" 2>&1
python tokenhsi/scripts/multi_agent/analyze_push_height_contacts.py --output "$ANALYSIS_OUT/hand_contacts" > "$ANALYSIS_OUT/hand_contacts.log" 2>&1
env -u HEIGHT_PD_QUICK python tokenhsi/scripts/multi_agent/analyze_push_height_pd.py \
 --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml \
 --cfg_env tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint.yaml \
 --motion_file tokenhsi/data/dataset_push_only_push180.yaml --num_agents 2 --num_envs 1488 --num_objects 4 \
 --task_graph push --test --eval --headless --no_video --output_path "$ANALYSIS_OUT/humanoid_pd" > "$ANALYSIS_OUT/humanoid_pd.log" 2>&1
python tokenhsi/scripts/multi_agent/summarize_push_height_analysis.py \
 --contacts "$ANALYSIS_OUT/hand_contacts" --pd "$ANALYSIS_OUT/humanoid_pd" \
 --forces "$ANALYSIS_OUT/known_forces" --output "$ANALYSIS_OUT"
