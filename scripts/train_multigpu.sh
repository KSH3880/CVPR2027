#!/bin/bash
# Data-parallel training across GPUs.
#
#   bash train_multigpu.sh 8 tokenhsi/scripts/single_task/traj_train.sh
#   bash train_multigpu.sh 4 tokenhsi/scripts/tokenhsi/stage1_train.sh
#
# Arg 1 is the GPU count, arg 2 is any *_train.sh from the README.
# Each rank runs its own simulator with --num_envs envs, so the effective
# batch is num_envs x GPU count. Only rank 0 prints stats and writes checkpoints.
#
# Defaults to the TokenHSI baseline. Pick another repo with TOKENHSI_REPO:
#   TOKENHSI_REPO=TokenHSI-steer bash train_multigpu.sh 4 tokenhsi/scripts/...
set -e

NPROC=${1:?usage: train_multigpu.sh <num_gpus> <train_script.sh> [extra args]}
SCRIPT=${2:?usage: train_multigpu.sh <num_gpus> <train_script.sh> [extra args]}
shift 2

REPO=${TOKENHSI_REPO:-TokenHSI}
[ -d /home/hwanhee/CVPR2027/$REPO/tokenhsi ] || { echo "no such repo: $REPO"; exit 1; }

source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi
cd /home/hwanhee/CVPR2027/$REPO

export PYTHONPATH=/home/hwanhee/CVPR2027/ddp_shim:$PYTHONPATH

# strip the leading "python ./tokenhsi/run.py" so torchrun can drive the script
ARGS=$(grep -v '^#!' "$SCRIPT" | tr '\n' ' ' | sed 's/\\/ /g' | sed 's|^ *python *\./tokenhsi/run\.py||')

torchrun --standalone --nproc_per_node=$NPROC ./tokenhsi/run.py $ARGS --horovod "$@"
