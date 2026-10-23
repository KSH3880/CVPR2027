#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=${TOKENHSI_GPU:-5}
export TOKENHSI_CONDA_ENV=tokenhsi
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/runtime_env.sh"
set -o pipefail
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi:$TOKENHSI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/stage1_push_midpoint_safe_build}

python "$TOKENHSI_ROOT/tokenhsi/tests/build_push_native_rsi.py" \
  --task HumanoidMAStage1Push \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_midpoint_safe.yaml \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml \
  --num_agents 2 --num_envs 2048 --num_objects 4 \
  --task_graph random_scenario --test --eval --headless --no_video \
  --output_path "$OUTPUT_PATH"

python - "$TOKENHSI_ROOT" "$OUTPUT_PATH/stage1_push_rect_native_rsi_cache.npz" <<'PY'
import hashlib
import re
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
destination = root / 'tokenhsi/data/stage1_push_midpoint_safe_cache.npz'
config = root / 'tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_safe.yaml'
if not source.is_file():
    raise SystemExit(f'RSI cache was not produced: {source}')
digest = hashlib.sha256(source.read_bytes()).hexdigest()
shutil.copy2(source, destination)
text = config.read_text()
text, count = re.subn(r'(rsi_cache_sha256:\s*)[0-9a-f]{64}', rf'\g<1>{digest}', text, count=1)
if count != 1:
    raise SystemExit('Could not update exactly one rsi_cache_sha256 field')
config.write_text(text)
print(f'Installed {destination}; sha256={digest}')
PY
