#!/bin/bash
# 돌고 있는 ma 학습의 **현재 정책**을 IsaacGym 뷰어로, VNC 로 본다.
# scripts/steer/view.sh 와 같은 방식이다 (Xvfb + x11vnc + noVNC).
#
#   PORT=6090 bash scripts/ma/view.sh t1_live_s0
#   PORT=6091 bash scripts/ma/view.sh t2_c05_b25 3
#   PORT=6092 MA_SEP=20 bash scripts/ma/view.sh t1_live_s0     간섭 없는 배치로
#   PORT=6093 MA_TOKEN=zero bash scripts/ma/view.sh t1_live_s0 토큰 끄고
#   RECORD=1 PORT=6090 bash scripts/ma/view.sh t1_live_s0      + mp4 저장
#
# 열기:
#   ssh -L <PORT>:localhost:<PORT> <host>
#   http://localhost:<PORT>/vnc.html
#
# 학습 프로세스에 붙는 게 아니다. 100 iter 마다 덮어쓰이는 체크포인트를 **복사해서**
# 별도로 띄운다 (쓰는 중인 파일을 열면 반쯤 쓰인 상태를 읽는다). 학습에는 영향 없다.
# 태그 대신 .pth 경로를 직접 줘도 된다.
set -e
ROOT=/home/hwanhee/CVPR2027
VNC_DIR=/home/hwanhee/opt/vnc
NOVNC_DIR=/home/hwanhee/opt/novnc

PORT=${PORT:-6090}
TAG=${1:?사용법: view.sh <tag 또는 ckpt경로> [env수]}
ENVS=${2:-3}

DISPNUM=$((PORT - 6059))
DISP=":$DISPNUM"
VNC_PORT=$((5900 + DISPNUM))

cd $ROOT/TokenHSI-ma
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

if [ -f "$TAG" ]; then CKPT="$TAG"; NAME=$(basename $(dirname $(dirname "$TAG")))
else
    CKPT=$(find output/ma/$TAG -name Humanoid.pth 2>/dev/null | head -1)
    [ -z "$CKPT" ] && { echo "체크포인트 없음: output/ma/$TAG"; exit 1; }
    NAME=$TAG
fi
SNAP=/tmp/ma_view_${NAME}_$PORT.pth; cp "$CKPT" "$SNAP"
ITER=$(grep -c "fps step" $ROOT/runs/queue/logs/$NAME.log 2>/dev/null || echo "?")
AGE=$(( $(date +%s) - $(stat -c %Y "$CKPT") ))

for p in $PORT $VNC_PORT; do
    ss -tln 2>/dev/null | grep -q ":$p " && { echo "포트 $p 사용 중 — PORT=$((PORT+1)) 로 다시"; exit 1; }
done
# 디스플레이도 검사한다. 포트만 보면 이미 떠 있는 남의 Xvfb 에 조용히 붙어서
# 두 시뮬이 한 화면에 겹쳐 나온다 (실제로 :31 에서 그랬다).
[ -e /tmp/.X11-unix/X$DISPNUM ] && { echo "디스플레이 $DISP 사용 중 — PORT=$((PORT+1)) 로 다시"; exit 1; }

cleanup() {
    if [ -n "${FFMPEG_PID:-}" ]; then kill -INT $FFMPEG_PID 2>/dev/null || true; wait $FFMPEG_PID 2>/dev/null || true; fi
    kill ${XVFB_PID:-} ${VNC_PID:-} ${WEB_PID:-} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CFG=$ROOT/runs/gen_cfgs/ma/view_${NAME}_$PORT.yaml
python3 - "$ENVS" "$CFG" <<'PY'
import re, sys, pathlib
s = pathlib.Path("tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml").read_text()
s = re.sub(r"^  numAgents:.*$", "  numAgents: 2", s, flags=re.M)
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {sys.argv[1]}", s, flags=re.M)
pathlib.Path(sys.argv[2]).write_text(s)
PY

Xvfb $DISP -screen 0 1600x900x24 -nolisten tcp & XVFB_PID=$!
sleep 2
LD_LIBRARY_PATH=$VNC_DIR/usr/lib/x86_64-linux-gnu $VNC_DIR/usr/bin/x11vnc \
    -display $DISP -rfbport $VNC_PORT -localhost -nopw -forever -shared -noxdamage -quiet & VNC_PID=$!
sleep 1
websockify --web=$NOVNC_DIR 127.0.0.1:$PORT 127.0.0.1:$VNC_PORT > /dev/null 2>&1 & WEB_PID=$!
sleep 1

if [ "${RECORD:-0}" != 0 ]; then
    VIDEO=$ROOT/runs/results/video/${NAME}_$(date +%H%M%S).mp4
    mkdir -p "$(dirname "$VIDEO")"
    # 왼쪽 323px 은 IsaacGym 패널이라 잘라낸다. RECORD_DELAY 는 모션 로딩(검은 화면)을 건너뛴다
    (sleep ${RECORD_DELAY:-90}; exec ffmpeg -nostdin -loglevel error -y \
        -f x11grab -draw_mouse 0 -framerate 30 -video_size 1276x900 -i $DISP+324,0 \
        -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$VIDEO") & FFMPEG_PID=$!
fi

echo "=============================================================="
echo " 열기     http://localhost:$PORT/vnc.html"
echo " 포워딩   ssh -L $PORT:localhost:$PORT $(hostname)"
echo " 태그     $NAME   iter≈$ITER   (${AGE}초 전 저장본)"
echo " env      $ENVS x 2명   MA_SEP=${MA_SEP:-0}   MA_TOKEN=${MA_TOKEN:-live}"
[ -n "${VIDEO:-}" ] && echo " 영상     $VIDEO"
echo "=============================================================="

DISPLAY=$DISP \
MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1} MA_TOKEN=${MA_TOKEN:-live} MA_SEP=${MA_SEP:-0} \
CUDA_VISIBLE_DEVICES=${MA_GPU:-4} \
python ./tokenhsi/run.py --task HumanoidMACarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint "$SNAP" \
    --test --eval_task carry
