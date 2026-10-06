#!/bin/bash
# Fetch allocation + its exact executors, then serve a local noVNC viewer.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/TokenHSI-coord/task_allocation"
if [ "${1:-}" = --help ]; then
    echo 'usage: bash task_allocation/scp_view.sh [SSH-host] [remote-allocation.pth]'
    echo 'defaults: Pro1 /home/hwanhee/juan/CVPR2027/runs/task_allocation/allocation_s0/allocation_000100.pth'
    echo 'overrides: MA_GPU=0 PORT=6109 CONDA_BASE=... TOKENHSI_CONDA_ENV=tokenhsi118'
    exit 0
fi
if [ "${1:-}" != --local ]; then
    HOST=${1:-Pro1}
    REMOTE=${2:-/home/hwanhee/juan/CVPR2027/runs/task_allocation/allocation_s0/allocation_000100.pth}
    IMPORT=$(mktemp -d "$ROOT/runs/task_allocation/import_vnc_XXXXXX")
    scp "$HOST:$REMOTE" "$IMPORT/allocation_original.pth"
    PYTHON_BIN="${CONDA_BASE:-/home/injesus1010/anaconda3}/envs/${TOKENHSI_CONDA_ENV:-tokenhsi118}/bin/python"
    [ -x "$PYTHON_BIN" ] || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
    "$PYTHON_BIN" - "$IMPORT" <<'PY'
import pathlib, shlex, sys, torch
p=pathlib.Path(sys.argv[1])
ck=torch.load(p/'allocation_original.pth',map_location='cpu',weights_only=False)
c=ck['config']
# Only checkpoint metadata supplies the remote executor filenames.
lines=[]
for key in ('executor','stage1'):
    value=c[key]
    if not isinstance(value,str) or not value.startswith('/') or '\n' in value:
        raise ValueError(f'invalid remote {key} path')
    lines.append(f'REMOTE_{key.upper()}={shlex.quote(value)}')
(p/'remote_paths.sh').write_text('\n'.join(lines)+'\n')
PY
    source "$IMPORT/remote_paths.sh"
    scp "$HOST:$REMOTE_EXECUTOR" "$IMPORT/ms18.pth"
    scp "$HOST:$REMOTE_STAGE1" "$IMPORT/stage1.pth"
    "$PYTHON_BIN" - "$IMPORT" <<'PY'
import json, pathlib, shlex, sys, torch
p=pathlib.Path(sys.argv[1]).resolve()
ck=torch.load(p/'allocation_original.pth',map_location='cpu',weights_only=False)
c=ck['config']
(p/'original_config.json').write_text(json.dumps(c,indent=2)+'\n')
# Preserve original bytes; relocate only executor paths in a separate viewer copy.
c['executor']=str(p/'ms18.pth'); c['stage1']=str(p/'stage1.pth')
torch.save(ck,p/'allocation_view.pth')
values=dict(ALLOC_INIT=str(p/'allocation_view.pth'),ALLOC_EXECUTOR=c['executor'],MS_CKPT=c['stage1'])
for key,env in dict(interval='INTERVAL',horizon='HORIZON',gamma='GAMMA',lam='LAMBDA',switch_coef='SWITCH_COEF',d_model='D_MODEL',extent='EXTENT',clearance='CLEARANCE',episode_steps='EPISODE_STEPS',seed='SEED').items():
    values['ALLOC_'+env]=c[key]
for env,value in zip(('TIME_COEF','DELIVERY_COEF','FAILURE_COEF'),c['costs']):
    values['ALLOC_'+env]=value
(p/'view.env').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k,v in values.items()))
PY
    export ALLOC_IMPORT="$IMPORT" MA_GPU=${MA_GPU:-0} PORT=${PORT:-6109}
    export NOVNC_LOCAL_VIEW="$HERE/scp_view.sh"
    echo "Downloaded files: $IMPORT"
    echo "VNC: http://localhost:$PORT/vnc.html"
    exec bash "$ROOT/scripts/masteer/view_sequential_stack.sh" --local
fi

: "${ALLOC_IMPORT:?run scp_view.sh without --local first}"
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
 --output_path "$ALLOC_OUTPUT/unused" > "$ALLOC_OUTPUT/run.log" 2>&1
