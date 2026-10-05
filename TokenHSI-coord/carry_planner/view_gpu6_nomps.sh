#!/bin/bash
# GPU6 noVNC Carry viewer, without an MPS client connection.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PLANNER=${1:-"$ROOT/runs/carry_planner/carry_implicit_curvature05_lr3e4_clip20_3000_s0/planner_000550.pth"}
export NOVNC_RUNTIME=${NOVNC_RUNTIME:-/home/hwanhee/opt/vnc}
export NOVNC_WEB=${NOVNC_WEB:-/home/hwanhee/opt/novnc}
export CONDA_BASE=${CONDA_BASE:-/home/hwanhee/anaconda3}
export TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi_juan}
export MS_CKPT=${MS_CKPT:-"/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth"}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}
export CARRY_PLANNER_VIEW_CPU_PHYSICS=${CARRY_PLANNER_VIEW_CPU_PHYSICS:-0}
export MA_GPU=6 PORT=${PORT:-6109}
export PYTHONDONTWRITEBYTECODE=1
exec bash "$ROOT/TokenHSI-coord/carry_planner/view_vnc.sh" "$PLANNER" \
 "$ROOT/TokenHSI-masteer/output/stack/ms18_maskteam_origscale_c06_s0_00009000.pth"
