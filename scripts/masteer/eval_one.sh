#!/bin/bash
# Final evaluation for one MA training row. Replays the row's saved environment
# and requires both the frozen stage-1 checkpoint and the trained adapt policy.
set -u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=${1:?usage: eval_one.sh <tag> <gpu> [envs]}
GPU=${2:?usage: eval_one.sh <tag> <gpu> [envs]}
ENVS=${3:-512}
# 중간 체크포인트 평가는 결과 tag와 원 학습 tag가 다르다.
# QUEUE_EVAL_SOURCE가 학습 env/cfg를, QUEUE_EVAL_CKPT가 정확한 가중치를 지정한다.
SOURCE=${QUEUE_EVAL_SOURCE:-$TAG}
# MS_EVAL_SUFFIX 는 호출자가 준다. 잠금·로그·지표를 여기서 갈라야 정상 평가와
# 대조군 평가가 서로 막지 않는다. 체크포인트와 cfg 는 원래 TAG 를 그대로 쓴다.
SUF=${MS_EVAL_SUFFIX:-}
if [ -n "$SUF" ]; then TAG_OUT="${TAG}__${SUF}"; else TAG_OUT="$TAG"; fi
CLAIM=$ROOT/runs/queue/gpu_locks/eval_$TAG_OUT
LOG=$ROOT/runs/queue/logs/eval_$TAG_OUT.log
ENV_FILE=$ROOT/runs/queue/logs/$SOURCE.env

mkdir -p "$ROOT/runs/queue/gpu_locks" "$ROOT/runs/queue/logs"
if ! mkdir "$CLAIM" 2>/dev/null; then
    echo "masteer eval: $TAG_OUT is already claimed" >&2
    exit 3
fi
trap 'rm -rf "$CLAIM"' EXIT
printf '%s\n' "$GPU" > "$CLAIM/gpu"

if [ ! -f "$ENV_FILE" ]; then
    echo "masteer eval: missing $ENV_FILE; refusing to guess training settings" >&2
    exit 4
fi
source "$ENV_FILE"

# 예전 sidecar는 미지정 값도 export NAME=''로 기록했다. 숫자 설정에서
# int('')/float('')가 되지 않도록 replay 시에는 원래 의미인 unset으로 복원한다.
while IFS= read -r name; do
    if [ -z "${!name}" ]; then
        unset "$name"
    fi
done < <(sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$ENV_FILE")

# 같은 정책을 다른 시나리오 설정으로 재평가하는 통로.
#
#   MS_EVAL_OVERRIDE="MS_DECEL=both MS_DT=2.0 MS_DT_RAND=0"
#   MS_EVAL_SUFFIX=both
#
# cross_both(둘 다 감속, 오프셋 0)와 cross_placebo(창을 교차점 뒤로)는 **같은 학습
# 정책의 대조군**이라 별도 학습 행이 아니다. 사이드카를 source 한 **뒤에** 덮어써야
# 한다 -- 앞에 두면 사이드카의 export 가 도로 밀어버린다.
if [ -n "${MS_EVAL_OVERRIDE:-}" ]; then
    for kv in $MS_EVAL_OVERRIDE; do
        case "$kv" in
            MS_*=*|MA_*=*) export "${kv?}" ;;
            *) echo "masteer eval: 거부된 override $kv" >&2; exit 6 ;;
        esac
    done
    echo "masteer eval: override $MS_EVAL_OVERRIDE"
fi

CK=${QUEUE_EVAL_CKPT:-}
if [ -z "$CK" ]; then
CK=$(python3 - "$ROOT/TokenHSI-masteer/output/masteer/$SOURCE" <<'PY'
import sys
from pathlib import Path

paths = list(Path(sys.argv[1]).glob("*/nn/Humanoid.pth"))
if paths:
    print(max(paths, key=lambda path: path.stat().st_mtime))
PY
)
fi
if [ -n "$CK" ] && [ ! -f "$CK" ]; then
    echo "masteer eval: checkpoint 없음: $CK" >&2
    exit 4
fi
TRAIN_ENV=$ROOT/runs/gen_cfgs/masteer/$SOURCE.yaml
if [ -z "$CK" ] || [ ! -f "$TRAIN_ENV" ]; then
    echo "masteer eval: missing checkpoint or generated env config for $SOURCE" >&2
    exit 4
fi

EVAL_ENV=$ROOT/runs/gen_cfgs/masteer/eval_$TAG.yaml
python3 - "$TRAIN_ENV" "$EVAL_ENV" "$ENVS" <<'PY'
import re, sys

text = open(sys.argv[1]).read()
text = re.sub(r"^  numEnvs:.*$", f"  numEnvs: {sys.argv[3]}", text, flags=re.M)
open(sys.argv[2], "w").write(text)
PY

TRAIN_CFG=${MS_TRAINCFG:-tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt.yaml}
if [ -f "$ROOT/runs/gen_cfgs/masteer/${SOURCE}_train.yaml" ]; then
    TRAIN_CFG=$ROOT/runs/gen_cfgs/masteer/${SOURCE}_train.yaml
fi
BASE_CKPT=${MS_CKPT:-output/tokenhsi/ckpt_stage1.pth}
METRICS=$ROOT/runs/results/masteer/eval_$TAG_OUT.npy
mkdir -p "$(dirname "$METRICS")"
rm -f "$METRICS"

cd "$ROOT/TokenHSI-masteer"
if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE=$("$CONDA_EXE" info --base)
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base)
    elif [ -f /home/cvlab/anaconda3/etc/profile.d/conda.sh ]; then
        CONDA_BASE=/home/cvlab/anaconda3
    else
        echo "masteer eval: conda를 찾지 못함" >&2
        exit 1
    fi
fi
set +u
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${TOKENHSI_CONDA_ENV:-tokenhsi}"
set -u
export CUDA_VISIBLE_DEVICES=$GPU
# **env 코드가 읽는 이름은 MA_METRICS 다.** MS_METRICS 만 내보내면 지표가 안 쓰인다.
export MS_METRICS=$METRICS
export MA_METRICS=$METRICS

python -u ./tokenhsi/run.py \
    --task "${MS_TASK:-HumanoidMASteerCarry}" \
    --cfg_train "$TRAIN_CFG" \
    --cfg_env "$EVAL_ENV" \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --num_envs "$ENVS" --headless --seed "${MS_SEED:-0}" \
    --hrl_checkpoint "$BASE_CKPT" --checkpoint "$CK" \
    --test --eval --eval_task "${MS_EVAL_TASK:-carry}" > "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -f "$METRICS" ]; then
    printf 'MS_EVAL_ERROR tag=%s rc=%s metrics=%s\n' "$TAG_OUT" "$rc" "$([ -f "$METRICS" ] && echo yes || echo no)" >> "$LOG"
    [ "$rc" -ne 0 ] && exit "$rc"
    exit 5
fi

python3 - "$TAG_OUT" "$LOG" "$METRICS" "${MS_AGENTS:-2}" >> "$LOG" 2>&1 <<'PY'
import re, sys
import numpy as np

tag, log_path, metrics_path, agents = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
text = open(log_path, errors="ignore").read()
rates = [float(value) for value in re.findall(r"'success_rate': ([0-9.]+)", text)]
stable = rates[1:] if len(rates) > 1 else rates
sr = sum(stable) / len(stable) if stable else float("nan")
m = np.load(metrics_path)
finished = m[:, 1] >= 0
agent = m[:, 0].astype(int) % agents
paths = [np.median(m[agent == index, 5]) for index in range(agents)]
fmt = lambda value: "NA" if not np.isfinite(value) else f"{value:.4f}"
# **주 요약 한 줄에 핵심 지표를 다 실어야 한다.** autofill 은 eval_summary_prefix 로
# 시작하는 한 줄만 원장에 남기므로, 별도 줄로 찍으면 AUTO_RESULTS 에서 사라진다.
head = (
    f"MS_EVAL_SUMMARY tag={tag} rc=0 sr={fmt(sr)} eps={len(m)} "
    f"fin={finished.mean():.4f} "
    f"t50={np.median(m[finished, 1]) if finished.any() else -1:.0f} "
    f"colEp={(m[:, 2] > 0).mean():.4f} colStep={m[:, 2].mean():.2f} "
    f"enter={m[:, 3].mean():.2f} still={m[:, 4].mean():.2f} "
    f"path={np.median(m[:, 5]):.2f} "
    f"pathA={','.join(f'{value:.2f}' for value in paths)}"
)

_med = lambda v: np.median(v) if len(v) else float("nan")
if m.shape[1] >= 31:                       # 경로 추종 -- free 기준선의 판정 근거
    _n = np.maximum(m[:, 30], 1)
    _lat, _spd, _vr = m[:, 26] / _n, m[:, 28] / _n, m[:, 29] / _n
    head += (f" lat={_med(_lat):.3f} spd_err={_med(_spd):.3f} v_real={_med(_vr):.3f} "
             f"off50={float((_lat > 0.5).mean()):.3f}")
    # **v_real 은 정지 스텝까지 분모에 넣어 보행속도를 심하게 낮춘다** (실측 0.58 대 1.02).
    # 그 값으로 "정책이 참조 모션보다 느리다" 고 잘못 판단했다. 이동 스텝만으로 다시 잰다.
    _mv = np.maximum(m[:, 30] - m[:, 4], 1)
    head += f" gait={_med(m[:, 5] / (_mv / 30.0)):.3f}"

# 열 40~49: 집기/운반 생애주기. grasp는 손-박스 거리 proxy이고 carry/place는
# 그 proxy 상태에서 박스를 0.5 m 이상 실제로 옮긴 뒤 목표에 닿았는지로 판정한다.
if m.shape[1] >= 50:
    _steps = np.maximum(m[:, 30], 1)
    head += (
        f" graspEp={(m[:, 44] >= 0).mean():.4f}"
        f" graspStep={np.median(m[:, 40] / _steps):.3f}"
        f" gateStep={np.median(m[:, 41] / _steps):.3f}"
        f" heldMove={np.median(m[:, 42]):.2f}"
        f" liftDz={np.median(m[:, 43]):.2f}"
        f" delivered={m[:, 45].mean():.4f}"
        f" putdown={m[:, 46].mean():.4f}"
        f" carry={m[:, 47].mean():.4f}"
        f" place={m[:, 48].mean():.4f}"
        f" shortcut={m[:, 49].mean():.4f}"
    )

# **쌍 단위 성공률.** fin 은 에이전트별이라 "한 명만 배달" 을 성공으로 센다.
# 실제 목표는 한 env 의 두 명이 **모두** 배달하는 것이고, 교차 시나리오에서
# 둘의 격차가 크다 (실측: 개별 0.720 인데 둘 다는 0.311, 독립 예상 0.518 의 60%).
# free 에서는 격차가 없어서(0.684 대 0.479 = 독립) 이 지표 없이는 교차의 방해를 못 본다.
if agents == 2:
    _env = m[:, 0].astype(int) // 2
    _f0 = dict(zip(_env[agent == 0], finished[agent == 0]))
    _f1 = dict(zip(_env[agent == 1], finished[agent == 1]))
    _k = sorted(set(_f0) & set(_f1))
    if _k:
        _b = np.array([[_f0[k], _f1[k]] for k in _k])
        _both = float((_b[:, 0] & _b[:, 1]).mean())
        head += (f" both={_both:.4f} one={float((_b[:, 0] ^ _b[:, 1]).mean()):.4f}"
                 f" indep={float(finished.mean() ** 2):.4f}")
    if m.shape[1] >= 39:
        _c = [0.375, 0.75, 1.125, 1.5]
        _xs, _ys = [], []
        for _i in range(4):
            _t = m[:, 35 + _i].sum()
            if _t > 0:
                _xs.append(_c[_i]); _ys.append(m[:, 31 + _i].sum() / _t)
        if len(_xs) > 1:
            head += f" vslope={np.polyfit(_xs, _ys, 1)[0]:.3f}"
            head += " vc=" + ",".join(f"{y:.2f}" for y in _ys)
if m.shape[1] >= 26 and int(m[0, 25]) != 0:   # 시나리오일 때만
    _env = m[:, 0].astype(int) // agents
    _wn, _wd, _dt = m[:, 16], m[:, 20], m[:, 19]
    _d0 = dict(zip(_env[agent == 0], _wn[agent == 0] / 30.0))
    _d1 = dict(zip(_env[agent == 1], _wn[agent == 1] / 30.0))
    _g0 = dict(zip(_env[agent == 0], _wd[agent == 0]))
    _g1 = dict(zip(_env[agent == 1], _wd[agent == 1]))
    _k = [k for k in sorted(set(_d0) & set(_d1)) if _g0.get(k, 0) > 0 and _g1.get(k, 0) > 0]
    if _k:
        _v = np.array([_d1[k] - _d0[k] for k in _k])
        _dte = dict(zip(_env[agent == 1], np.round(_dt[agent == 1], 2)))
        head += f" dTw={_med(_v):+.3f} wpass={len(_k)/max(len(set(_d0)),1):.3f}"
        _lv = np.unique([_dte.get(k, 0.0) for k in _k])
        if len(_lv) > 1:
            _xs = [L for L in _lv]
            _ys = [_med(_v[[abs(_dte.get(k, 0.0) - L) < 1e-6 for k in _k]]) for L in _lv]
            _ok = [i for i, y in enumerate(_ys) if np.isfinite(y)]
            if len(_ok) > 1:
                head += f" beta={np.polyfit([_xs[i] for i in _ok], [_ys[i] for i in _ok], 1)[0]:.3f}"
    _ed = m[:, 15]
    head += f" encd={_med(_ed[_ed < 98]):.3f}"
print(head)

# 늘 재는 것: 경로 이탈과 속도 추종. **free 기준선이야말로 이게 있어야 한다** --
# 성공률만으로는 "경로를 따라간 것" 과 "목표로 직진한 것" 을 구분할 수 없다.
if m.shape[1] >= 31:
    med = lambda v: np.median(v) if len(v) else float("nan")
    n = np.maximum(m[:, 30], 1)
    latr, latb, spd, vr = m[:, 26]/n, m[:, 27]/n, m[:, 28]/n, m[:, 29]/n
    # 명령 속도별 실제 속도 -- steer 주장의 직접 증거.
    # 낮은 명령만 따르고 높은 것을 못 따르면 곡선이 평평해진다.
    curve = ""
    if m.shape[1] >= 39:
        cmds = [0.375, 0.75, 1.125, 1.5]
        parts = []
        for i in range(4):
            tot = m[:, 35 + i].sum()
            if tot > 0:
                parts.append(f"{cmds[i]:.3f}->{m[:, 31 + i].sum()/tot:.3f}")
        if parts:
            curve = " v곡선[" + " ".join(parts) + "]"
    line = (f"MS_TRACK_SUMMARY tag={tag} "
            f"lat_root={med(latr):.3f} lat_box={med(latb):.3f} "
            f"spd_err={med(spd):.3f} v_real={med(vr):.3f} "
            f"이탈0.5초과={float((latr > 0.5).mean()):.3f} 이탈1.0초과={float((latr > 1.0).mean()):.3f}")
    for index in range(agents):
        line += f" a{index}lat={med(latr[agent == index]):.3f}"
    line += curve
    print(line)

# 시나리오 열 13~25. train.sh 와 **같은 정의**를 쓴다.
# 주 지표는 env 당 쌍 dT_w (창 통과 시간차). 중앙값끼리 빼면 집기 산포가
# 잡음으로 다시 들어오므로 반드시 쌍으로 본다.
if m.shape[1] >= 26:
    DT = 1.0 / 30.0
    med = lambda v: np.median(v) if len(v) else float("nan")
    env = m[:, 0].astype(int) // agents

    def pair(vals):
        d0 = dict(zip(env[agent == 0], vals[agent == 0]))
        d1 = dict(zip(env[agent == 1], vals[agent == 1]))
        keys = sorted(set(d0) & set(d1))
        return np.array(keys), np.array([d1[k] - d0[k] for k in keys])

    wn, wdone, dtc = m[:, 16], m[:, 20], m[:, 19]
    xt, xd, ed, latw, encn = m[:, 13], m[:, 14], m[:, 15], m[:, 22], m[:, 23]
    line = f"MS_SCEN_SUMMARY tag={tag} scen={int(m[0, 25])}"

    w0 = dict(zip(env[agent == 0], wdone[agent == 0]))
    w1 = dict(zip(env[agent == 1], wdone[agent == 1]))
    keys, dTw = pair(wn * DT)
    # 창을 **끝까지** 지난 쌍만. 안 그러면 "느려서 오래" 와 "넘어짐" 이 섞인다.
    sel = np.array([w0.get(k, 0) > 0 and w1.get(k, 0) > 0 for k in keys]) if len(keys) else np.array([])
    if sel.any():
        line += f" dTw={med(dTw[sel]):+.3f} pass={sel.mean():.3f} n={int(sel.sum())}"
        dte = dict(zip(env[agent == 1], np.round(dtc[agent == 1], 2)))
        levels = np.unique([dte.get(k, 0.0) for k in keys[sel]])
        if len(levels) > 1:
            xs, ys, parts = [], [], []
            for L in levels:
                mk = np.array([abs(dte.get(k, 0.0) - L) < 1e-6 for k in keys[sel]])
                if mk.any():
                    v = med(dTw[sel][mk]); parts.append(f"{L:.1f}:{v:+.3f}")
                    xs.append(L); ys.append(v)
            if len(xs) > 1:
                line += f" beta={np.polyfit(xs, ys, 1)[0]:.3f}"
            line += " curve=" + ",".join(parts)
        # 공간평균 속도. 시간평균은 느린 프레임에 가중이 실려 아래로 편향된다.
        for index in range(agents):
            w = (agent == index) & (wdone > 0) & (wn > 0)
            line += f" a{index}v={3.0/max(med(wn[w])*DT, 1e-6):.3f}" if w.any() else f" a{index}v=NA"
        # 확인 지표: 교차점 도달 시각차. **양쪽 다** 닿은 쌍만.
        x0 = dict(zip(env[agent == 0], xd[agent == 0]))
        x1 = dict(zip(env[agent == 1], xd[agent == 1]))
        kx, dtx = pair(xt * DT)
        sx = np.array([x0.get(k, 99) < 1.0 and x1.get(k, 99) < 1.0 for k in kx]) if len(kx) else np.array([])
        if sx.any():
            line += f" dtx={med(dtx[sx]):+.3f} gate={sx.mean():.3f}"
        lw = wn > 0
        line += (f" encd={med(ed[ed < 98]):.3f} encn={med(encn):.0f} "
                 f"latw={med(latw[lw]/np.maximum(wn[lw], 1)):.3f}")
    else:
        line += " (창 통과 쌍 없음)"
    print(line)
PY

grep '^MS_EVAL_SUMMARY ' "$LOG" | tail -1
grep '^MS_SCEN_SUMMARY ' "$LOG" | tail -1
grep '^MS_TRACK_SUMMARY ' "$LOG" | tail -1
