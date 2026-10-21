#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_ROOT=$(cd "$ROOT/.." && pwd)
COMPARE=${1:-pair}
case "$COMPARE" in pair|none_zero|hide_humans|hide_doors|hide_boxes|hide_goals|hide_boxes_goals|hide_objects_goals|hide_box_first|hide_box_second|remove_doors) ;; *) echo 'Usage: stage1_native4_attention_ablation.sh [pair|none_zero|hide_humans|hide_doors|hide_boxes|hide_goals|hide_boxes_goals|hide_objects_goals|hide_box_first|hide_box_second|remove_doors]' >&2; exit 2;; esac
export NATIVE4_ATTENTION_COMPARE=$COMPARE
OUT=${OUTPUT_PATH:-$PROJECT_ROOT/runs/jointpush/native4_attention_${COMPARE}_$(date +%Y%m%d_%H%M%S)}
case "$OUT" in /*) ;; *) OUT="$PROJECT_ROOT/$OUT";; esac
mkdir -p "$OUT"
for scope in full "$COMPARE"; do
  echo "Attention scope=$scope output=$OUT/$scope"
  NATIVE4_ATTENTION_SCOPE=$scope OUTPUT_PATH="$OUT/$scope" \
    bash "$ROOT/scripts/stage1_native4_zeroshot.sh" eval > "$OUT/$scope.log" 2>&1
done
python - "$OUT" "$COMPARE" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
intervention = sys.argv[2]
full, pair = [json.loads((out / mode / 'result.json').read_text()) for mode in ('full', intervention)]
assert full['initial_state_sha256'] == pair['initial_state_sha256'], 'Initial worlds differ'
result = dict(initial_states_identical=True, intervention=intervention, full_fall_rate=full['scene_fall_rate'],
              full_preproximity_fall_scenes=full['scenes_with_preproximity_fall'],
              first_step_action=full['first_step_action'], scenes=full['scenes'],
              horizon_steps=full['horizon_steps'])
result[intervention + '_fall_rate'] = pair['scene_fall_rate']
result[intervention + '_preproximity_fall_scenes'] = pair['scenes_with_preproximity_fall']
(out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
