#!/bin/bash
# ma 트랙의 한 슬롯. autofill 이 큐 표의 환경변수와 함께 이걸 부른다.
#
#   MS_MODE=sweep   짧게 돌려 PhysX 요구량과 fps 만 본다 (격자)
#   MS_MODE=test    체크포인트로 평가만 (학습 없음)
#   MS_MODE=train   stage1 에서 전체 fine-tune (평범한 빌더). --resume 필요
#   MS_MODE=adapt   adapt 스캐폴드 (backbone 동결 + 학습 토큰). --hrl_checkpoint
#
#   MS_TASK  태스크 클래스. 토큰 실험은 HumanoidMACarry
#   MA_TOKEN live|zero|r12|r18|mask
#     mask 는 live 관측을 유지하되 teammate 토큰을 attention 에서 정확히 제외한다
#
# 통과 조건은 rc=0 이 아니라 warn=0 이다. 경고가 뜨는 설정이 더 빠른데, 그건 접촉을
# 버려서 빠른 것이다 (docs/ENV_SCALING.md).
#
# 이 파일을 고칠 때는 새로 쓰고 mv 로 갈아끼운다. 실행 중인 bash 는 스크립트를 증분으로
# 읽으므로 제자리 덮어쓰기는 돌고 있는 런을 죽인다 (전례: unexpected EOF 로 6 런 사망).
set -u
cd /home/hwanhee/CVPR2027/TokenHSI-masteer
source /home/hwanhee/anaconda3/etc/profile.d/conda.sh
conda activate tokenhsi

TAG=${MS_TAG:?MS_TAG is required}
MODE=${MS_MODE:-sweep}
ENVS=${MS_ENVS:-2048}
AGENTS=${MS_AGENTS:-1}
CP=${MS_CP:-8}                 # max_gpu_contact_pairs, M 단위 (8 = 8M)
ITERS=${MS_ITERS:-14}
SEED=${MS_SEED:-0}
MB=${MS_MB:-0}                 # minibatch. 0 이면 원본 유지.
                               # num_envs 를 바꾸면 이것도 같은 배수로 움직여야
                               # 롤아웃당 업데이트 수가 유지된다 (ENV_SCALING §1)
CKPT=${MS_CKPT:-output/tokenhsi/ckpt_stage1.pth}
# 지표는 env 안에서 스텝마다 누적하고 주기적으로 여기에 쓴다 (사후 계산 금지, PITFALLS #6)
# **env 코드가 읽는 이름은 MA_METRICS 다.** MS_METRICS 만 export 하던 탓에
# masteer 지표가 한 번도 안 쓰였다 (runs/results/masteer/ 가 계속 비어 있었다).
export MS_METRICS=${MS_METRICS:-/home/hwanhee/CVPR2027/runs/results/masteer/${TAG}.npy}
export MA_METRICS=$MS_METRICS
export MA_TAU MA_STILL_V MA_SEP MA_SPAWN_GAP MA_LAYOUT MA_LAYOUT_L MA_LAYOUT_D MA_LAYOUT_S
export MA_C MA_BETA MA_K MA_TOKEN MA_TOKENIZER_ZERO MA_TEAM MA_MKSPN MA_TREF MA_NOFREEZE
export MA_TRAJ MA_TRAJ_ENVS MA_TRAJ_STEPS MA_DHIST MA_DHIST_STEPS
export MS_K MS_M_NOM MS_M_LO MS_MRAND MS_POS_C MS_REWARD_OUTER MS_BACK MS_PIN MS_ZERO MS_SEED
export MS_LAT_MAX MS_TURN_MAX MS_HUMP2 MS_SKEW MS_SPREAD_MIN MS_SPREAD_MAX MS_LAT_FRAC
# 시나리오. MS_SCEN 하나가 배치 변수들을 일관되게 정한다 (env 쪽 _scen_env)
export MS_SCEN MS_GAP MS_DT MS_W MS_L MS_SEP MS_ENC_R
# 보상·명령 knob. 여기 빠지면 값을 줘도 안 먹는데 **아무 오류도 안 난다** --
# MS_CLIP 이 실제로 그렇게 조용히 무시됐다 (arc_end 가 안 잘려서 발견).
export MS_VEL_W MS_VEL_K MS_ENDCLAMP MS_CLIP MS_DT_RAND MS_DT_SET MS_DECEL MS_PLACEBO MS_RECOV MS_DBG MS_GRADCHK
mkdir -p "$(dirname "$MS_METRICS")"

CFG_SRC=tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml
CFG=/home/hwanhee/CVPR2027/runs/gen_cfgs/masteer/${TAG}.yaml
mkdir -p "$(dirname "$CFG")"

# 슬롯마다 자기 cfg 사본을 만든다. 돌고 있는 파일을 고치면 다른 슬롯이 같이 바뀐다.
python3 - "$CFG_SRC" "$CFG" "$AGENTS" "$CP" "$ENVS" ${MS_SPACING:+"$MS_SPACING"} <<'PY'
import re, sys
src, dst, agents, cp, envs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
s = open(src).read()
s = re.sub(r"^  numAgents:.*$", f"  numAgents: {agents}", s, flags=re.M)
# rl_games 의 num_actors 는 여기서 온다. --num_envs 로 덮어도 따라오지 않는다.
s = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {envs}", s, flags=re.M)
if len(sys.argv) > 6:
    s = re.sub(r"^  envSpacing:.*$", f"  envSpacing: {sys.argv[6]}", s, flags=re.M)
s = re.sub(r"^(\s*)default_buffer_size_multiplier:",
           rf"\1max_gpu_contact_pairs: {cp * 1024 * 1024}\n\1default_buffer_size_multiplier:",
           s, count=1, flags=re.M)
open(dst, "w").write(s)
PY

# adapt 는 별도 학습 설정을 쓴다 (동결·adapt mlp 설정이 거기 있다)
TRAIN_SRC=${MS_TRAINCFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml}
TRAIN_CFG=$TRAIN_SRC
if [ "$MB" != "0" ]; then
    TRAIN_CFG=/home/hwanhee/CVPR2027/runs/gen_cfgs/masteer/${TAG}_train.yaml
    sed "s/^\( *\)minibatch_size:.*/\1minibatch_size: $MB/" "$TRAIN_SRC" > "$TRAIN_CFG"
fi

COMMON=(--task "${MS_TASK:-HumanoidMASteerCarry}"
        --cfg_train "$TRAIN_CFG"
        --cfg_env "$CFG"
        --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml
        --num_envs "$ENVS" --headless --seed "$SEED")

case "$MODE" in
sweep)  python -u ./tokenhsi/run.py "${COMMON[@]}" --hrl_checkpoint "$CKPT" \
            --max_iterations "$ITERS" --output_path "output/masteer/$TAG" ;;
test)   python -u ./tokenhsi/run.py "${COMMON[@]}" --checkpoint "$CKPT" \
            --test --eval --eval_task "${MA_EVAL_TASK:-carry}" ;;
# stage1 에서 이어 학습하려면 --resume 1 과 --checkpoint 가 **둘 다** 필요하다.
# --resume 가 load_checkpoint 를 켜고 --checkpoint 가 경로를 준다 (utils/config.py:125-129).
# --checkpoint 만 주면 복원이 조용히 건너뛰어져 **랜덤 초기화로 학습한다** (실측: 1000 iter -> 0.0).
# --hrl_checkpoint 는 stage2 합성 태스크가 stage1 을 별도 low-level 정책으로 쓸 때의 것이라 여기선 틀리다.
train)
        # --resume 는 epoch 카운터까지 복원한다. stage1 은 40,000 epoch 이라
        # --max_iterations 를 그대로 주면 "MAX EPOCHS NUM!" 으로 즉시 끝난다
        # (실측: 6000 을 줬더니 1 iter 만에 종료). MS_ITERS 는 "여기서 몇 번 더" 다.
        base=$(python3 -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu').get('epoch',0))" "$CKPT")
        echo "[ma] resume epoch=$base  +${ITERS}  -> max_iterations $((base + ITERS))"
        python -u ./tokenhsi/run.py "${COMMON[@]}" --checkpoint "$CKPT" --resume 1 \
            --max_iterations "$((base + ITERS))" --output_path "output/masteer/$TAG" ;;
adapt)
        # adapt 는 --hrl_checkpoint 로 stage1 을 **동결 부분에** 싣는다. rl_games 의
        # epoch 카운터를 복원하지 않으므로 MS_ITERS 가 곧 실제 학습량이다
        # (train 모드처럼 40,000 을 더할 필요가 없다).
        #
        # MA_RESUME=1 : 죽은 런을 이어서 돌린다. --hrl_checkpoint(동결부, stage1) 는
        # 그대로 두고 --checkpoint 로 **자기 학습본**을 얹은 뒤 --resume 으로 epoch 을
        # 복원한다. 그러면 1900 에서 죽은 런이 MS_ITERS(3000)까지 이어진다.
        RES=""
        if [ "${MA_RESUME:-0}" = "1" ]; then
            # 가장 최근 것을 고른다. 같은 태그를 여러 번 돌리면 디렉터리가 여러 개 생기는데
            # head -1 은 순서가 보장되지 않아 옛 체크포인트를 잡을 수 있다.
            OWN=$(find "output/masteer/$TAG" -name Humanoid.pth -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d" " -f2-)
            if [ -n "$OWN" ]; then
                cp "$OWN" "/tmp/resume_$TAG.pth"
                RES="--checkpoint /tmp/resume_$TAG.pth --resume 1"
                echo "[ma] resume $TAG <- $OWN"
            else
                echo "[ma] resume 요청됐지만 체크포인트 없음 -> 처음부터"
            fi
        fi
        python -u ./tokenhsi/run.py "${COMMON[@]}" --hrl_checkpoint "$CKPT" $RES \
            --max_iterations "$ITERS" --output_path "output/masteer/$TAG" ;;
atest)  python -u ./tokenhsi/run.py "${COMMON[@]}" --hrl_checkpoint "$CKPT" \
            --test --eval --eval_task "${MA_EVAL_TASK:-carry}" ;;
*)      echo "unknown MS_MODE=$MODE"; exit 1 ;;
esac
rc=$?

# grep -c 는 0 건일 때도 "0" 을 찍고 종료코드 1 을 낸다. `|| echo 0` 을 붙이면 "0" 이
# 두 번 나와 값에 개행이 섞이고 뒤 필드가 통째로 밀린다. || true 여야 한다.
LOG=/home/hwanhee/CVPR2027/runs/queue/logs/${TAG}.log
warn=$(grep -c foundLostAggregatePairs "$LOG" 2>/dev/null || true)
need=$(grep -o 'Capacity to [0-9]*' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
fps=$(grep -oE 'fps total: [0-9.]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
# 평가는 0 회차를 버린다 -- 리셋이 안 가라앉은 상태라 값이 다르다 (PITFALLS #5)
sr=$(grep -oE "'success_rate': [0-9.]+" "$LOG" 2>/dev/null | grep -oE '[0-9.]+' |
     awk 'NR>1{s+=$1;n++} END{if(n)printf "%.4f",s/n}')
# 정상 상태 처리량: 앞쪽 절반을 버린다 (기동·워밍업이 섞인다)
sfps=$(grep -oE 'fps step: [0-9.]+' "$LOG" | grep -oE '[0-9.]+$' |
       awk '{a[NR]=$1} END{n=asort(a); s=0; c=0; for(i=int(NR/2);i<=NR;i++){s+=a[i];c++} if(c)printf "%.0f", s/c}' 2>/dev/null ||
       grep -oE 'fps step: [0-9.]+' "$LOG" | grep -oE '[0-9.]+$' | tail -n +8 |
       awk '{s+=$1;c++} END{if(c)printf "%.0f", s/c}')
met=$(python3 - "$MS_METRICS" 2>/dev/null <<'PYX'
import sys, numpy as np
try:
    m = np.load(sys.argv[1])
except Exception:
    sys.exit()
f, ok = m[:, 1], m[:, 1] >= 0
out = (f"eps={m.shape[0]} fin={ok.mean():.3f} "
       f"t50={np.median(f[ok]) if ok.any() else -1:.0f} "
       f"colEp={(m[:,2]>0).mean():.3f} dmin={np.median(m[:,6]):.2f} "
       f"path={np.median(m[:,5]):.2f}")
if m.shape[1] >= 26:
    # **env 당 쌍으로 본다.** 한 env 의 두 행은 같은 시계·같은 조우를 공유하므로
    # 중앙값끼리 빼면 집기 시간 산포가 그대로 잡음으로 다시 들어온다.
    DT = 1.0 / 30.0
    a = m[:, 0].astype(int) % 2
    env = m[:, 0].astype(int) // 2
    med = lambda v: np.median(v) if len(v) else float("nan")
    def pair(vals):
        d0 = dict(zip(env[a == 0], vals[a == 0]))
        d1 = dict(zip(env[a == 1], vals[a == 1]))
        k = sorted(set(d0) & set(d1))
        return np.array(k), np.array([d1[i] - d0[i] for i in k])
    wn, wdone, dtc = m[:, 16], m[:, 20], m[:, 19]
    xt, xd, ed, latw, encn = m[:, 13], m[:, 14], m[:, 15], m[:, 22], m[:, 23]

    # 주 지표: 창 통과 시간차. **창을 끝까지 지난 쌍만** -- 안 그러면
    # "느려서 오래 걸림" 과 "중간에 넘어짐" 이 같은 값으로 섞인다.
    w0 = dict(zip(env[a == 0], wdone[a == 0]))
    w1 = dict(zip(env[a == 1], wdone[a == 1]))
    ke, dTw = pair(wn * DT)
    sel = np.array([w0.get(e, 0) > 0 and w1.get(e, 0) > 0 for e in ke]) if len(ke) else np.array([])
    if sel.any():
        out += f" | dTw={med(dTw[sel]):+.3f}s pass={sel.mean():.2f} n={int(sel.sum())}"
        dte = dict(zip(env[a == 1], np.round(dtc[a == 1], 2)))
        lv = np.unique([dte.get(e, 0.0) for e in ke[sel]])
        if len(lv) > 1:
            xs, ys, parts = [], [], []
            for L in lv:
                mk = np.array([abs(dte.get(e, 0.0) - L) < 1e-6 for e in ke[sel]])
                if mk.any():
                    v = med(dTw[sel][mk]); parts.append(f"{L:.1f}:{v:+.2f}")
                    xs.append(L); ys.append(v)
            if len(xs) > 1:
                out += f" beta={np.polyfit(xs, ys, 1)[0]:.3f}"
            out += " [" + " ".join(parts) + "]"
    # 공간평균 속도 W/(wn*dt). 시간평균은 느린 프레임에 가중이 실려 아래로 편향된다.
    for k in (0, 1):
        w = (a == k) & (wdone > 0) & (wn > 0)
        out += f" a{k}v={3.0/max(med(wn[w])*DT, 1e-6):.3f}" if w.any() else f" a{k}v=NA"
    # 확인 지표: 교차점 도달 시각차. **양쪽 다** 교차점에 닿은 쌍만 쓴다.
    d0 = dict(zip(env[a == 0], xd[a == 0])); d1 = dict(zip(env[a == 1], xd[a == 1]))
    kx, dtx = pair(xt * DT)
    sx = np.array([d0.get(e, 99) < 1.0 and d1.get(e, 99) < 1.0 for e in kx]) if len(kx) else np.array([])
    if sx.any():
        out += f" | dtx={med(dtx[sx]):+.3f}s gate={sx.mean():.2f}"
    lw = wn > 0
    out += (f" encd={med(ed[ed < 98]):.2f} encn={med(encn):.0f} "
            f"latw={med(latw[lw] / np.maximum(wn[lw], 1)):.3f}")
print(out)
PYX
)
printf 'MA_METRIC tag=%s %s\n' "$TAG" "${met:-none}"
printf 'MA_SUMMARY tag=%s agents=%s envs=%s cp=%sM mb=%s seed=%s rc=%s warn=%s need=%s fps=%s sfps=%s sr=%s\n' \
    "$TAG" "$AGENTS" "$ENVS" "$CP" "$MB" "$SEED" "$rc" "${warn:-0}" "${need:-0}" "${fps:-NA}" "${sfps:-NA}" "${sr:-NA}"
exit $rc
