#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${KSH_PARTITION_WORK:-$ROOT/runs/jointpush/ksh_partition_depth180_9000_20261022/workspace}
SOURCE=${KSH_PUSH_SOURCE:-/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push}
PYTHON=${KSH_PARTITION_PYTHON:-/home/hwanhee/anaconda3/envs/tokenhsi_juan/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export TOKENHSI_CONDA_ENV=tokenhsi_juan
. "$ROOT/jointpush/ports/koo_native_y2m/tokenhsi/scripts/multi_agent/runtime_env.sh"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$ROOT/runs/jointpush/ksh_partition_torch_extensions"
export CXX=/usr/bin/g++-11
CHECKPOINT=${STAGE1_CHECKPOINT:-$SOURCE/output/stage1_depth180_20kg_push100_from1000/Stage1FiveTaskDepth18020kg_21-20-09-05/nn/Stage1FiveTaskDepth18020kg_00009000.pth}
cd "$WORK"
args=()
if [ "${HEADLESS:-0}" = 1 ]; then args+=(--headless); fi
exec "$PYTHON" "$ROOT/stage2_jointpush/scripts/ksh_partition_run.py" --task HumanoidMAKshPartitionPush \
 --cfg_env "$WORK/configs/env.yaml" --cfg_train "$WORK/configs/train.yaml" \
 --motion_file tokenhsi/data/dataset_five_task.yaml --checkpoint "$CHECKPOINT" \
 --num_agents 4 --num_objects 8 --num_envs "${NUM_ENVS:-1}" --task_graph push_at \
 --eval_skills loco --eval_skill_probs 1.0 --eval_repeats "${REPEATS:-1000000}" \
 --output_path "${OUTPUT_PATH:-$WORK/output_etc/check/view}" --test --eval --no_video --seed "${SEED:-2141}" \
 --sim_device cuda:0 --rl_device cuda:0 "${args[@]}"
