#!/usr/bin/env bash
# Native-size payload inference: [asset=teamhoi_default_rectangle] [model=anchor]
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
source "$ROOT/tokenhsi/scripts/multi_agent/runtime_env.sh"
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY=''
unset CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE TOKENHSI_INFERENCE_CONFIG
export JOINT_ZERO_SHOT_GPU_PHYSX=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
ASSET=${1:-teamhoi_default_rectangle}
MODEL=${2:-anchor}
if [[ "${EVAL_SKILLS:-loco}" != loco ]]; then
 echo 'Native-size furniture requires loco start; box carryWith/pickUp/putDown RSI is incompatible.' >&2
 exit 2
fi
mapfile -t PATHS < <(python - "$ASSET" "$MODEL" <<'PY'
from pathlib import Path
import json, sys
asset, model = sys.argv[1:]
models = json.loads(Path('SA4_SOURCE.json').read_text())['models']
if model not in models:
    raise SystemExit('model must be basic or anchor')
root = Path('../runs/jointcarry_zeroshot/native_assets').resolve()
names = {item['name'] for item in json.loads((root/'catalog.json').read_text())}
if asset not in names and asset != 'box':
    raise SystemExit('Unknown native asset: '+asset)
m = models[model]
for key in ('checkpoint','train_config','env_config'):
    p = Path(m[key]).resolve(strict=True)
    print(p)
if asset == 'box':
    print(''); print('')
else:
    print((root/asset/'asset.urdf').resolve(strict=True))
    print((root/asset/'manifest.json').resolve(strict=True))
PY
)
[[ ${#PATHS[@]} == 5 ]] || { echo 'Failed to resolve model/asset paths' >&2; exit 2; }
if [[ "$ASSET" == box ]]; then
 unset JOINT_ZERO_SHOT_URDF JOINT_ZERO_SHOT_NATIVE_CONFIG
else
 export JOINT_ZERO_SHOT_URDF=${PATHS[3]} JOINT_ZERO_SHOT_NATIVE_CONFIG=${PATHS[4]}
fi
OUT=${OUTPUT_PATH:-$ROOT/../runs/jointcarry_sa4/native_${MODEL}_${ASSET}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
python - "$MODEL" "$ASSET" "$OUT" <<'PY'
import json, sys, hashlib, os
from pathlib import Path
model, asset, out = sys.argv[1:]
m = json.loads(Path('SA4_SOURCE.json').read_text())['models'][model]
assert hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest() == m['sha256']
metadata = dict(model=model, asset=asset, checkpoint=m, initial_skill='loco', task_graph='joint_carry_at', physical_gpu=6)
if os.environ.get('JOINT_ZERO_SHOT_NATIVE_CONFIG'):
    metadata['native_asset'] = json.loads(Path(os.environ['JOINT_ZERO_SHOT_NATIVE_CONFIG']).read_text())
Path(out,'native_run.json').write_text(json.dumps(metadata,indent=2)+'\n')
PY
VIEW_FLAGS=(--headless)
if [[ "${HEADLESS:-1}" == 0 ]]; then VIEW_FLAGS=(--task_camera stack); fi
python tokenhsi/scripts/prepare_joint_carry.py
python tokenhsi/run.py \
 --task HumanoidMACarry --cfg_train "${PATHS[1]}" --cfg_env "${PATHS[2]}" \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "${PATHS[0]}" \
 --num_agents 2 --num_envs "${NUM_ENVS:-1}" --num_objects 4 \
 --eval_repeats "${EVAL_REPEATS:-1}" --eval_skills loco --eval_skill_probs 1.0 \
 --task_graph joint_carry_at --episode_length "${EPISODE_LENGTH:-600}" \
 --seed "${SEED:-20261016}" --sim_device cuda:0 --pipeline gpu --rl_device cuda:0 \
 "${VIEW_FLAGS[@]}" --no_video --test --eval --output_path "$OUT" 2>&1 | tee "$OUT/eval.log"
