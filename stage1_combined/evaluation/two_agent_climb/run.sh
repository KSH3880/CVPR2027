#!/usr/bin/env bash
set -eo pipefail
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTORCH_JIT=0 PYTHONUNBUFFERED=1
. "$(dirname "$0")/../../tokenhsi/scripts/multi_agent/runtime_env.sh"
set -u
CATEGORY=$1
OUT=${2:-output_etc/check/two_agent_climb_78k_20261025}
if [ -f "$OUT/$CATEGORY/results.json" ]; then
  echo "Existing completed result: $OUT/$CATEGORY; choose a fresh output directory." >&2
  exit 2
fi
export CLIMB_CATEGORY=$CATEGORY
python evaluation/two_agent_climb/launch.py --task TwoAgentClimb \
 --cfg_train "$OUT/train.yaml" --cfg_env "$OUT/env.yaml" \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
 --checkpoint checkpoints/Stage1V2Fixed_00078000.pth \
 --num_agents 2 --num_objects 4 --num_envs 512 --task_graph climb \
 --test --eval --eval_skills loco --eval_skill_probs 1.0 --eval_repeats 3 \
 --headless --seed 0 --output_path "$OUT/$CATEGORY" --no_video
