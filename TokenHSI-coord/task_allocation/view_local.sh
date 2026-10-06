#!/bin/bash
# View only: local checkpoints, no scp and no training.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?new-tag required}
INITIAL=${2:?allocation.pth required}
POLICY=${3:-"$ROOT/TokenHSI-masteer/output/ms18_maskteam_origscale_c06_s0_00009000.pth"}
STAGE1=${MS_CKPT:-"$ROOT/TokenHSI-masteer/output/ckpt_stage1.pth"}
case "$TAG" in ""|*[!A-Za-z0-9_.-]*) echo 'invalid tag' >&2; exit 2;; esac
for file in "$INITIAL" "$POLICY" "$STAGE1"; do
    [ -f "$file" ] || { echo "checkpoint missing: $file" >&2; exit 2; }
done
ALLOC_IMPORT="$ROOT/runs/task_allocation/$TAG"
[ ! -e "$ALLOC_IMPORT" ] || { echo "use a new tag: $ALLOC_IMPORT" >&2; exit 2; }
PYTHON_BIN="${CONDA_BASE:-/home/injesus1010/anaconda3}/envs/${TOKENHSI_CONDA_ENV:-tokenhsi118}/bin/python"
[ -x "$PYTHON_BIN" ] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$ALLOC_IMPORT"
"$PYTHON_BIN" - "$ALLOC_IMPORT" "$INITIAL" "$POLICY" "$STAGE1" <<'PY'
import json, pathlib, shlex, sys, torch
p=pathlib.Path(sys.argv[1]).resolve()
ck=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
c=ck['config']
(p/'original_config.json').write_text(json.dumps(c,indent=2)+'\n')
# Preserve original bytes; relocate only executor paths in a separate viewer copy.
c['executor']=str(pathlib.Path(sys.argv[3]).resolve()); c['stage1']=str(pathlib.Path(sys.argv[4]).resolve())
torch.save(ck,p/'allocation_view.pth')
values=dict(ALLOC_INIT=str(p/'allocation_view.pth'),ALLOC_EXECUTOR=c['executor'],MS_CKPT=c['stage1'])
for key,env in dict(interval='INTERVAL',horizon='HORIZON',gamma='GAMMA',lam='LAMBDA',switch_coef='SWITCH_COEF',d_model='D_MODEL',extent='EXTENT',clearance='CLEARANCE',episode_steps='EPISODE_STEPS',seed='SEED').items():
    values['ALLOC_'+env]=c[key]
for env,value in zip(('TIME_COEF','DELIVERY_COEF','FAILURE_COEF'),c['costs']):
    values['ALLOC_'+env]=value
(p/'view.env').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k,v in values.items()))
PY
source "$ALLOC_IMPORT/view.env"
GPU=${MA_GPU:-0}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'MA_GPU must be a GPU index' >&2; exit 2; }
ENVS=${ALLOC_ENVS:-1}
[[ "$ENVS" =~ ^[1-9][0-9]*$ ]] || { echo 'ALLOC_ENVS must be positive' >&2; exit 2; }
CONDA_ROOT=${CONDA_BASE:-/home/injesus1010/anaconda3}
ENV_NAME=${TOKENHSI_CONDA_ENV:-tokenhsi118}
export PATH="$CONDA_ROOT/envs/$ENV_NAME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
# This local viewer uses a separate, absent MPS endpoint; shared daemons stay running.
unset CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
export CUDA_MPS_PIPE_DIRECTORY="$ROOT/runs/tools/allocation-view-no-mps"
[ ! -e "$CUDA_MPS_PIPE_DIRECTORY/control" ] && [ ! -e "$CUDA_MPS_PIPE_DIRECTORY/nvidia-cuda-mps-control.pid" ] || { echo 'viewer isolation path has an MPS endpoint' >&2; exit 2; }
export TORCH_EXTENSIONS_DIR="$ROOT/runs/cache/torch_extensions" CUDA_CACHE_PATH="$ROOT/runs/cache/cuda"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID"
export ALLOC_MODE=eval ALLOC_ENVS="$ENVS" ALLOC_ITERS=${ALLOC_ITERS:-10000}
export ALLOC_OUTPUT="$ALLOC_IMPORT/view_output"
mkdir -p "$ALLOC_OUTPUT"
EXEC_REPO="$ROOT/TokenHSI-masteer"
python "$ROOT/TokenHSI-coord/carry_box_swap/configure.py" \
 "$EXEC_REPO/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml" \
 "$ALLOC_OUTPUT/env.yaml" --envs "$ENVS" --skill loco_carry --steps "$ALLOC_EPISODE_STEPS"
export CARRY_PLANNER_PHYSICAL_GPU="$GPU" CARRY_BOX_SWAP_GOALS_FOLLOW_BOX=1
export COORD_PROVIDER=external COORD_MODEL=c1 COORD_DRAW_CANDIDATES=0 COORD_VIEWER=0
export CARRY_PLANNER_CONVERGE_PROB=0 CARRY_PLANNER_RANDOMIZE_AGENT_SLOTS=0
export MA_TOKEN=mask MA_TOKENIZER_ZERO=1 MS_MRAND=0 MS_M_LO=0 MS_CLIP=1 MS_ZERO=0
export MS_SCEN=cross MS_SEED="$ALLOC_SEED" MS_REWARD_OUTER=1 MS_POS_C=0.6 MS_VEL_W=1 MS_VIEW_TIMED_CROSS=0
export MS_DRAW_TASK=1 COORD_PRESERVE_PICKUP_APPROACH=0 COORD_ALLOW_HAND_CONTACT=1 STACK_CURRICULUM=0
export MA_SEP=0 MA_SPAWN_GAP=0 MA_C=0 MA_BETA=0 MA_TEAM=share MA_MKSPN=0 MA_TAU=.3
unset MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_S MA_LAYOUT_D MS_SCEN_CURVE MA_METRICS MA_DHIST MS_METRICS
source "$ROOT/TokenHSI-coord/stack_planner/physx_cuda_compat.sh"
echo "Viewer GPU $GPU; log: $ALLOC_OUTPUT/run.log"
cd "$ROOT/TokenHSI-coord"
python -u -m task_allocation.train --test --graphics_device_id "${TOKENHSI_GRAPHICS_DEVICE_ID:-$GPU}" \
 --task HumanoidTaskAllocationMS18 --sim_device cuda:0 --rl_device cuda:0 --physx --pipeline gpu \
 --cfg_train "$EXEC_REPO/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml" \
 --cfg_env "$ALLOC_OUTPUT/env.yaml" --motion_file "$EXEC_REPO/tokenhsi/data/dataset_loco_sit_carry_climb.yaml" \
 --hrl_checkpoint "$MS_CKPT" --checkpoint "$ALLOC_EXECUTOR" --num_envs "$ENVS" --seed "$ALLOC_SEED" \
 --output_path "$ALLOC_OUTPUT/unused" 2>&1 | tee "$ALLOC_OUTPUT/run.log"
