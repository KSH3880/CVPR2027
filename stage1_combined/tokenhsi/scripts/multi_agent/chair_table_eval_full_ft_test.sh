#!/usr/bin/env bash
export TOKENHSI_GPU=${TOKENHSI_GPU:-0} TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} PYTHONDONTWRITEBYTECODE=1
# CUDA 11.8 NVRTC cannot target RTX 5090; eager kernels work on this host.
export PYTORCH_JIT=${PYTORCH_JIT:-0}
. "$(dirname "$0")/runtime_env.sh"
set -eu
if [ "${HEADLESS:-1}" = 0 ]; then . "$TOKENHSI_ROOT/tokenhsi/scripts/multi_agent/gui_gpu_env.sh"; fi
CKPT=$(realpath "$1")
export OUTPUT_PATH=${OUTPUT_PATH:-output_etc/check/chair_table_eval_ft_test_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTPUT_PATH"
python - "$CKPT" "$OUTPUT_PATH/env.yaml" <<'PY'
from pathlib import Path
import sys,torch,yaml
weights=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
source=weights['relation_experiment_config'];cfg={'env':source['env'],'sim':source['sim']}
cfg['env']['chairFT']['asset_split']='test'
Path(sys.argv[2]).write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
set -- --task_graph "${TASK_GRAPH:-sit_climb}"
if [ -n "${EPISODE_LENGTH:-}" ]; then set -- "$@" --episode_length "$EPISODE_LENGTH"; fi
if [ "${HEADLESS:-1}" != 0 ]; then set -- "$@" --headless; fi
python tokenhsi/run.py --task HumanoidChairFT \
 --cfg_train tokenhsi/data/cfg/train/rlg/chair_table_eval_full_ft.yaml --cfg_env "$OUTPUT_PATH/env.yaml" \
 --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml --checkpoint "$CKPT" \
 --num_agents 2 --num_envs "${CHAIR_TEST_ENVS:-16}" --num_objects 4 \
 --test --eval --eval_repeats "${CHAIR_REPEATS:-1}" --eval_skills loco --eval_skill_probs 1.0 \
 --output_path "$OUTPUT_PATH" --no_video --seed 42 "$@"
