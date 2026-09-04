#!/bin/bash
# HumanoidMAStopCarry checkpoint viewer over noVNC.
#
# Formal floor-pickup parity (default: MA_SEP=9, carryWith=0, eval 3 repeats):
#   MSTOP_GPU=6 PORT=6109 ENVS=1 \
#     bash scripts/mastop/view.sh mastop_b_from_a6k_s0
# Set MSTOP_VIEW_EVAL=0 for a continuous viewer.
#
# Open locally after SSH forwarding:
#   ssh -L 6100:localhost:6100 <host>
#   http://localhost:6100/vnc.html
set -euo pipefail

ROOT=/home/hwanhee/koo_cvpr
REPO=$ROOT/TokenHSI-mastop
VNC_DIR=/home/hwanhee/opt/vnc
NOVNC_DIR=/home/hwanhee/opt/novnc

PORT=${PORT:-6100}
TAG=${1:-mastop_b_from_a6k_s0}
ENVS=${ENVS:-1}
GPU=${MSTOP_GPU:-${CUDA_VISIBLE_DEVICES:-6}}
case "$GPU" in
    6|7) ;;
    *) echo "MSTOP_GPU must be physical GPU 6 or 7" >&2; exit 2 ;;
esac

# CUDA_VISIBLE_DEVICES only remaps CUDA. Isaac Gym's viewer also initializes
# Vulkan, which otherwise selects physical GPU 0 independently. On this server,
# Mesa's PCI selector does not filter the NVIDIA devices, while the numeric
# selector does. Expose only the requested physical card to Vulkan, then address
# it as graphics ordinal 0 inside the process. The guard verifies device count
# and UUID before Xvfb or Isaac Gym starts.
GPU_DRI=$(python3 "$ROOT/scripts/vulkan_gpu_guard.py" "$GPU")

DISPNUM=$((PORT - 6059))
DISP=:$DISPNUM
VNC_PORT=$((5900 + DISPNUM))

cd "$REPO"
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi_koo

CFG=${MSTOP_VIEW_CFG:-output/mastop_carry/configs/${TAG}.yaml}
TRAIN_CFG=output/mastop_carry/configs/${TAG}_train.yaml
ENV_FILE=${MSTOP_VIEW_ENV:-$ROOT/runs/logs/mastop_carry/${TAG}.env}
if [ ! -f "$CFG" ] || [ ! -f "$TRAIN_CFG" ]; then
    echo "config 없음: $CFG 또는 $TRAIN_CFG" >&2
    exit 2
fi
if [ ! -f "$ENV_FILE" ]; then
    echo "학습 sidecar 없음: $ENV_FILE" >&2
    exit 2
fi

VIEW_CARRYWITH=${MSTOP_CARRYWITH:-0}
VIEW_SEP=${MA_SEP:-9}
VIEW_MIX=${MSTOP_MIX:-}
VIEW_STAGE=${MSTOP_STAGE:-}
VIEW_EVAL=${MSTOP_VIEW_EVAL:-1}
VIEW_TASK=${MSTOP_VIEW_TASK:-HumanoidMAStopCarry}
source "$ENV_FILE"
export MSTOP_CARRYWITH=$VIEW_CARRYWITH
export MA_SEP=$VIEW_SEP
export MSTOP_GPU=$GPU
export MSTOP_DIAG=0
unset MSTOP_DIAG_PATH MSTOP_DIAG_EVERY MA_METRICS
if [ -n "$VIEW_MIX" ]; then
    export MSTOP_MIX=$VIEW_MIX
fi
if [ -n "$VIEW_STAGE" ]; then
    export MSTOP_STAGE=$VIEW_STAGE
fi

CKPT=$(find "output/mastop_carry/$TAG" -name Humanoid.pth -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | sed -n '1s/^[^ ]* //p')
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "체크포인트 없음: output/mastop_carry/$TAG" >&2
    exit 2
fi

for p in "$PORT" "$VNC_PORT"; do
    if ss -tln 2>/dev/null | grep -q ":$p "; then
        echo "포트 $p 사용 중 - 다른 PORT로 다시 실행" >&2
        exit 3
    fi
done
if [ -e "/tmp/.X11-unix/X$DISPNUM" ]; then
    echo "디스플레이 $DISP 사용 중 - 다른 PORT로 다시 실행" >&2
    exit 3
fi

mkdir -p "$ROOT/runs/view_snapshots"
SNAP=$ROOT/runs/view_snapshots/mastop_view_${TAG}_${PORT}.pth
cp "$CKPT" "$SNAP"

cleanup() {
    kill "${XVFB_PID:-}" "${VNC_PID:-}" "${WEB_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

Xvfb "$DISP" -screen 0 1600x900x24 -nolisten tcp & XVFB_PID=$!
sleep 2
LD_LIBRARY_PATH=$VNC_DIR/usr/lib/x86_64-linux-gnu \
    $VNC_DIR/usr/bin/x11vnc -display "$DISP" -rfbport "$VNC_PORT" \
    -localhost -nopw -forever -shared -noxdamage -noshm -quiet & VNC_PID=$!
sleep 1
websockify --web="$NOVNC_DIR" "127.0.0.1:$PORT" "127.0.0.1:$VNC_PORT" \
    >/dev/null 2>&1 & WEB_PID=$!
sleep 1
case "$VIEW_EVAL" in
    0) EVAL_ARGS=() ;;
    1) EVAL_ARGS=(--eval --eval_task carry) ;;
    *) echo "MSTOP_VIEW_EVAL must be 0 or 1" >&2; exit 2 ;;
esac



echo "=============================================================="
echo " 열기       http://localhost:$PORT/vnc.html"
echo " 포워딩     ssh -L $PORT:localhost:$PORT $(hostname)"
echo " 태그       $TAG"
echo " task       $VIEW_TASK"
echo " env cfg    $CFG"
echo " sidecar    $ENV_FILE"
echo " 체크포인트 $CKPT"
echo " GPU        physical $GPU (CUDA logical 0, Vulkan $GPU_DRI)"
echo " env        $ENVS x 2명"
echo " 시나리오   carry stage=$MSTOP_STAGE mix=$MSTOP_MIX carryWith=$MSTOP_CARRYWITH"
echo " 배치/eval  MA_SEP=$MA_SEP eval=$VIEW_EVAL"
echo "=============================================================="

DISPLAY=$DISP \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select DRI_PRIME=$GPU_DRI \
MS_CAM=${MS_CAM:-top} MS_CAM_H=${MS_CAM_H:-17} MS_CAM_B=${MS_CAM_B:-9} \
MSTOP_SEED=${MSTOP_SEED:-0} \
python -u ./tokenhsi/run.py \
    --task "$VIEW_TASK" \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "${MSTOP_BASE_CKPT:-$ROOT/TokenHSI/output/tokenhsi/ckpt_stage1.pth}" \
    --checkpoint "$SNAP" \
    --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
    --test "${EVAL_ARGS[@]}" --num_envs "$ENVS" --seed "${MSTOP_SEED:-0}"
