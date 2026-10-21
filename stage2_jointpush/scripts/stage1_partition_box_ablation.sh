#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
source "$ROOT/ports/koo_push/tokenhsi/scripts/multi_agent/runtime_env.sh"
PACKAGE="$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/native4_zeroshot"
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/partition_stage1_boxes_$(date +%Y%m%d_%H%M%S)}
case "$OUT" in /*) ;; *) OUT="$PROJECT_ROOT/$OUT";; esac
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /usr/bin/g++-11 ]; then export CXX=${TOKENHSI_CXX:-/usr/bin/g++-11}; fi
export PYTHONUNBUFFERED=1 PROBE_AGENTS=4 PROBE_STEPS=${PROBE_STEPS:-300}
for ordinary in 4 3 2 1 0; do
  scope=keep_ordinary_$ordinary
  if [ "$ordinary" = 4 ]; then scope=full; fi
  SOURCE=$(python "$PACKAGE/partition_stage1_setup.py" "$OUT/$scope" "${STAGE1_CHECKPOINT:-}")
  comparison=$scope
  if [ "$scope" = full ]; then comparison=keep_ordinary_0; fi
  echo "Partition Stage1 ordinary=$ordinary scope=$scope"
  NATIVE4_ATTENTION_SCOPE=$scope NATIVE4_ATTENTION_COMPARE=$comparison \
  PROBE_OUT="$OUT/$scope/result.json" \
  python "$PACKAGE/partition_stage1_launch.py" --task HumanoidMAPartitionPush \
    --cfg_env "$OUT/$scope/env.yaml" --cfg_train "$OUT/$scope/train.yaml" \
    --motion_file "${PARTITION_PUSH_MOTION_FILE:-tokenhsi/data/partition_push_motions.yaml}" \
    --num_agents 4 --num_objects 8 --num_envs "${NUM_ENVS:-64}" \
    --output_path "$OUT/$scope/evaluation" --experiment PartitionStage1ZeroShot \
    --no_video --task_graph push --test --eval --headless --eval_repeats 1 \
    --eval_skills loco --eval_skill_probs 1.0 --checkpoint "$SOURCE" --seed "${SEED:-123}" \
    > "$OUT/$scope.log" 2>&1
done
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1]); rows = []; hashes = set()
for ordinary in (4, 3, 2, 1, 0):
    name = 'full' if ordinary == 4 else 'keep_ordinary_' + str(ordinary)
    d = json.loads((out / name / 'result.json').read_text())
    hashes.add(d['initial_state_sha256'])
    rows.append(dict(ordinary_box_tokens=ordinary, total_box_tokens=4 + ordinary,
                     falls=sum(t['scene_fall'] for t in d['trials']), scenes=d['scenes'],
                     preproximity_fall_scenes=d['scenes_with_preproximity_fall']))
assert len(hashes) == 1
result = dict(initial_states_identical=True, physical_doors=0, original_stage1=True,
              physical_boxes_unchanged=True, rows=rows)
(out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
