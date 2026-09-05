#!/bin/bash
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?사용법: train_handclear_footpen_local.sh <tag> <foot_box_weight>}
FOOT_W=${2:?사용법: train_handclear_footpen_local.sh <tag> <foot_box_weight>}

export MA_GPU=6
export MA_INIT_CKPT="$ROOT/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth"
export MA_FREEZE_NEW_CARRY=0
export MA_TOKEN=mask

export STACK_HAND_ONLY_SWITCH=1
export STACK_HAND_CLEAR=0.10
export STACK_HAND_STEPS=5
export STACK_BODY_CLEAR=0.60
export STACK_CLEAR_HARD_GATE=0
export STACK_DROP_XY=100.0
export STACK_FOOT_CLEAR=0.20
export STACK_FOOT_BOX_W="$FOOT_W"
export STACK_CLEAR_STEER_W=0.25

exec bash "$ROOT/scripts/masteer/train_stack_release_local.sh" "$TAG"
