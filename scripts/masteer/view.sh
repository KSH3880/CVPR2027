#!/bin/bash
# 돌고 있는 masteer 학습의 **현재 정책**을 IsaacGym 뷰어로, VNC 로 본다.
# scripts/steer/view.sh 와 같은 방식이다 (Xvfb + x11vnc + noVNC).
#
#   PORT=6100 bash scripts/masteer/view.sh t1_live_s0
#   PORT=6091 bash scripts/masteer/view.sh t2_c05_b25 3
#   PORT=6092 MS_SCEN=cross MS_DT=2.0 bash scripts/masteer/view.sh ms4_cross_s0
#   PORT=6093 MS_SCEN=cross MS_DT=0   bash scripts/masteer/view.sh ms4_cross_s0   대조군
#   PORT=6094 MS_SCEN=parallel MS_GAP=1.0 bash scripts/masteer/view.sh ms4_par10_s0
#
# **시나리오 변수를 안 주면 free(랜덤 경로)로 뜬다.** 태그가 cross 로 학습된 것이어도
# 그렇다 -- 화면은 멀쩡한데 다른 조건을 보게 되므로 반드시 같이 준다.
#   RECORD=1 PORT=6100 bash scripts/masteer/view.sh t1_live_s0      + mp4 저장
#
# **MS_CLIP 기본값은 1 이다.** ms11 이후 계열은 전부 MS_CLIP=1 로 학습됐고,
# 이 값은 창·조준점·래칫을 바꾸는 **관측** 노브라 학습과 다르면 정책이 다른 입력을
# 본다 (MS_VEL_W·MS_ENDCLAMP 는 보상 전용이라 추론에는 무관하다).
# MS_CLIP=0 으로 학습한 옛 태그(ms1~ms10)를 볼 때만 MS_CLIP=0 을 명시한다.
#
# 시나리오를 그냥 얹어보고 싶으면 MS_DBG=0 을 준다. 기본값 1 은 배치 기하가
# 의도와 다르면 소리내어 죽는다 -- 지금 cross/parallel/solo 는 경로가 휘어서
# 반드시 걸린다. MS_DBG=0 이면 휜 채로 그냥 돈다 (구경은 된다).
#   PORT=6103 MS_DBG=0 MS_SCEN=cross MS_DT=2.0 bash scripts/masteer/view.sh <tag>
#
# 열기:
#   ssh -L <PORT>:localhost:<PORT> <host>
#   http://localhost:<PORT>/vnc.html
#
# 학습 프로세스에 붙는 게 아니다. 100 iter 마다 덮어쓰이는 체크포인트를 **복사해서**
# 별도로 띄운다 (쓰는 중인 파일을 열면 반쯤 쓰인 상태를 읽는다). 학습에는 영향 없다.
# 태그 대신 .pth 경로를 직접 줘도 된다.
set -e
ROOT=/home/hwanhee/juan/CVPR2027
VNC_DIR=/home/hwanhee/opt/vnc
NOVNC_DIR=/home/hwanhee/opt/novnc

PORT=${PORT:-6100}
TAG=${1:?사용법: view.sh <tag 또는 ckpt경로> [env수]}
# **ENVS 환경변수를 우선한다.** 예전에는 위치인자 $2 만 봐서 `ENVS=1 view.sh tag 3`
# 이 조용히 3 으로 떴다. record.sh 는 처음부터 ENVS 를 쓰므로 여기에 맞춘다.
# 위치인자는 그대로 남겨 둔다 (ENVS 를 안 주면 $2, 그것도 없으면 3).
ENVS=${ENVS:-${2:-3}}

# MS_VIZ 를 구체 변수로 편다. record.sh 와 **같은 파일**을 쓴다.
. "$ROOT/scripts/masteer/viz_env.sh"
viz_expand || exit 1

DISPNUM=$((PORT - 6059))
DISP=":$DISPNUM"
VNC_PORT=$((5900 + DISPNUM))

cd $ROOT/TokenHSI-masteer
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi_juan

export LD_LIBRARY_PATH=/tmp/hwanhee-physx-lib:${LD_LIBRARY_PATH:-}

if [ "${MS_BASE_ONLY:-0}" = "1" ]; then
    # MA 학습 checkpoint를 사용하지 않고 MS_CKPT만 로드한다.
    NAME=$(basename "$TAG" .pth)
    SNAP=""
    ITER="base-only"
    AGE=0
else
    if [ -f "$TAG" ]; then
        CKPT="$TAG"
        NAME=$(basename "$(dirname "$(dirname "$TAG")")")
    else
        CKPT=$(find "output/masteer/$TAG" -name Humanoid.pth 2>/dev/null | head -1)
        [ -z "$CKPT" ] && {
            echo "체크포인트 없음: output/masteer/$TAG"
            exit 1
        }
        NAME=$TAG
    fi

    SNAP=/tmp/ma_view_${NAME}_$PORT.pth
    cp "$CKPT" "$SNAP"
    ITER=$(grep -c "fps step" "$ROOT/runs/queue/logs/$NAME.log" 2>/dev/null || echo "?")
    AGE=$(( $(date +%s) - $(stat -c %Y "$CKPT") ))
fi

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

CFG=$ROOT/runs/gen_cfgs/masteer/view_${NAME}_$PORT.yaml
python3 - "$ENVS" "$CFG" "${MS_AGENTS:-2}" <<'PY'
import pathlib
import re
import sys

envs = sys.argv[1]
cfg_path = sys.argv[2]
agents = sys.argv[3]

src = pathlib.Path(
    "tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml"
)
s = src.read_text()

s = re.sub(
    r"^  numAgents:.*$",
    f"  numAgents: {agents}",
    s,
    flags=re.M,
)
s = re.sub(
    r"^  numEnvs:.*$",
    f"  numEnvs: {envs}",
    s,
    flags=re.M,
)
s = re.sub(
    r"^(\s*)default_buffer_size_multiplier:",
    r"\1max_gpu_contact_pairs: 4194304\n"
    r"\1default_buffer_size_multiplier:",
    s,
    count=1,
    flags=re.M,
)

pathlib.Path(cfg_path).write_text(s)
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
echo " 시나리오 ${MS_VIZ:-(직접지정)}"
 echo " env      $ENVS x 2명   배치=${MS_SCEN:-free}  지연=${MS_DT:-0}s  속도구간=${MS_MRAND:-0}  곡률=${MS_LAT_MAX:-2.2}  CLIP=${MS_CLIP:-1}"
[ -n "${VIDEO:-}" ] && echo " 영상     $VIDEO"
echo "=============================================================="

PLAYER_CKPT=()
if [ "${MS_BASE_ONLY:-0}" != "1" ]; then
    PLAYER_CKPT=(--checkpoint "$SNAP")
fi

# **시나리오 변수를 넘겨야 한다.** 안 넘기면 태그가 cross 로 학습된 것이어도
# 뷰어는 free(랜덤 경로)로 뜬다 -- 화면은 멀쩡한데 다른 조건을 보게 된다.
export DISPLAY="$DISP"
export CUDA_VISIBLE_DEVICES="${MA_GPU:-4}"
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

MA_TOKENIZER_ZERO=${MA_TOKENIZER_ZERO:-1} MA_TOKEN=${MA_TOKEN:-live} MA_SEP=${MA_SEP:-0} \
MA_SPAWN_GAP=${MA_SPAWN_GAP:-1.0} MS_MRAND=${MS_MRAND:-4} MS_ZERO=${MS_ZERO:-0} \
MS_SCEN=${MS_SCEN:-free} MS_DT=${MS_DT:-0} MS_DT_RAND=${MS_DT_RAND:-0} \
MS_DT_SET=${MS_DT_SET:-0,0.5,1.0,1.5,2.0} MS_DECEL=${MS_DECEL:-a1} \
MS_L=${MS_L:-12} MS_RECOV=${MS_RECOV:-1.5} MS_W=${MS_W:-3.0} MS_GAP=${MS_GAP:-1.0} \
MS_SEP=${MS_SEP:-9.0} MS_PLACEBO=${MS_PLACEBO:-0} \
MS_VEL_W=${MS_VEL_W:-1} MS_VEL_K=${MS_VEL_K:-5} \
MS_ENDCLAMP=${MS_ENDCLAMP:-0} MS_CLIP=${MS_CLIP:-1} MS_DBG=${MS_DBG:-1} \
python ./tokenhsi/run.py --task "${MS_TASK:-HumanoidMASteerCarry}" \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml \
    --cfg_env "$CFG" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint "${MS_CKPT:-output/tokenhsi/ckpt_stage1.pth}" \
    "${PLAYER_CKPT[@]}" \
    --test --eval_task carry
