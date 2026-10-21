#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
source "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/runtime_env.sh"
PACKAGE="$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/native4_zeroshot"
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/native2_no_doors_$(date +%Y%m%d_%H%M%S)}
case "$OUT" in /*) ;; *) OUT="$PROJECT_ROOT/$OUT";; esac
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /usr/bin/g++-11 ]; then export CXX=${TOKENHSI_CXX:-/usr/bin/g++-11}; fi
export PYTHONUNBUFFERED=1 PROBE_AGENTS=2 PROBE_SPAWN=native
export NATIVE4_ATTENTION_COMPARE=remove_doors PROBE_STEPS=${PROBE_STEPS:-300}
for scope in full remove_doors; do
  SOURCE=$(python "$PACKAGE/setup.py" "$OUT/$scope" "${STAGE1_CHECKPOINT:-}" --agents 2)
  echo "Native2 scope=$scope output=$OUT/$scope"
  NATIVE4_ATTENTION_SCOPE=$scope PROBE_OUT="$OUT/$scope/result.json" \
  python "$PACKAGE/native2_launch.py" --task HumanoidMAOpenPush \
    --cfg_env "$OUT/$scope/env.yaml" --cfg_train "$OUT/$scope/train.yaml" \
    --motion_file "${PARTITION_PUSH_MOTION_FILE:-tokenhsi/data/partition_push_motions.yaml}" \
    --num_agents 2 --num_objects 4 --num_envs "${NUM_ENVS:-64}" \
    --output_path "$OUT/$scope/evaluation" --experiment Stage1Native2 \
    --no_video --task_graph push --test --eval --headless --eval_repeats 1 \
    --eval_skills loco --eval_skill_probs 1.0 --checkpoint "$SOURCE" --seed "${SEED:-123}" \
    > "$OUT/$scope.log" 2>&1
done
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
full, reduced = [json.loads((out / name / 'result.json').read_text()) for name in ('full', 'remove_doors')]
assert full['initial_state_sha256'] == reduced['initial_state_sha256']
result = dict(initial_states_identical=True, agents=2, physical_boxes=2,
              physical_doors_retained=True, scenes=full['scenes'], horizon_steps=full['horizon_steps'],
              full_fall_rate=full['scene_fall_rate'], remove_doors_fall_rate=reduced['scene_fall_rate'],
              first_step_action=full['first_step_action'])
(out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
