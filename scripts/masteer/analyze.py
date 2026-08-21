#!/usr/bin/env python3
"""masteer 평가 npy 를 한 장으로 읽는다.

성공률만으로는 "경로를 따라간 것" 과 "목표로 직진한 것" 이 구분되지 않으므로
경로 추종(lat)·속도 추종(gait 곡선)·타이밍(ΔT_w)을 같이 낸다.

주의: v_real(열 29/30) 은 정지 스텝까지 분모에 넣어 보행속도를 낮게 보이게 한다.
      보행속도는 path/(이동 스텝) 으로 따로 잰다 (실측 0.58 대 1.02).
"""
import sys, glob, os
import numpy as np

DT = 1.0 / 30.0
med = lambda v: float(np.median(v)) if len(v) else float("nan")


def load(tag):
    p = f"runs/results/masteer/eval_{tag}.npy"
    return np.load(p) if os.path.exists(p) else None


def basic(m, A=2):
    fin = m[:, 1] >= 0
    steps = np.maximum(m[:, 30], 1) if m.shape[1] > 30 else np.full(len(m), 600.0)
    move = np.maximum(steps - m[:, 4], 1)
    d = dict(n=len(m), fin=fin.mean(), t50=med(m[fin, 1]), path=med(m[:, 5]),
             still=med(m[:, 4] / steps), colEp=(m[:, 2] > 0).mean(),
             dmin=med(m[:, 6]), gait=med(m[:, 5] / (move / 30.0)))
    if m.shape[1] > 30:
        n = steps
        d["lat"] = med(m[:, 26] / n)
        d["latbox"] = med(m[:, 27] / n)
        d["spd_err"] = med(m[:, 28] / n)
        d["off50"] = float((m[:, 26] / n > 0.5).mean())
    return d


def vcurve(m):
    """명령 속도별 실제 보행속도. 기울기 1.0 이면 완전 추종, 0 이면 무시."""
    if m.shape[1] < 39:
        return None
    cmds, xs, ys = [0.375, 0.75, 1.125, 1.5], [], []
    for i in range(4):
        t = m[:, 35 + i].sum()
        if t > 0:
            xs.append(cmds[i]); ys.append(m[:, 31 + i].sum() / t)
    if len(xs) < 2:
        return None
    return dict(x=xs, y=ys, slope=float(np.polyfit(xs, ys, 1)[0]))


def timing(m, A=2):
    """창 통과 시간차를 env 당 쌍으로. 중앙값끼리 빼면 집기 산포가 다시 들어온다."""
    if m.shape[1] < 26 or int(m[0, 25]) == 0:
        return None
    a, env = m[:, 0].astype(int) % A, m[:, 0].astype(int) // A
    wn, wd, dtc = m[:, 16] * DT, m[:, 20], np.round(m[:, 19], 2)
    d0, d1 = dict(zip(env[a == 0], wn[a == 0])), dict(zip(env[a == 1], wn[a == 1]))
    g0, g1 = dict(zip(env[a == 0], wd[a == 0])), dict(zip(env[a == 1], wd[a == 1]))
    dte = dict(zip(env[a == 1], dtc[a == 1]))
    ks = [k for k in sorted(set(d0) & set(d1)) if g0.get(k, 0) > 0 and g1.get(k, 0) > 0]
    if len(ks) < 30:
        return dict(n=len(ks), note="창 통과 쌍 부족")
    v = np.array([d1[k] - d0[k] for k in ks])
    L = np.array([dte.get(k, 0.0) for k in ks])
    lv = sorted(set(L))
    out = dict(n=len(ks), pass_rate=len(ks) / max(len(set(d0)), 1),
               curve=[(x, med(v[L == x])) for x in lv],
               encd=med(m[:, 15][m[:, 15] < 98]),
               latw=med(m[:, 22][m[:, 16] > 0] / np.maximum(m[:, 16][m[:, 16] > 0], 1)))
    if len(lv) > 1:
        out["beta"] = float(np.polyfit(lv, [c[1] for c in out["curve"]], 1)[0])
    return out


TAGS = sys.argv[1:] or [os.path.basename(p)[5:-4]
                        for p in sorted(glob.glob("runs/results/masteer/eval_ms*.npy"))]
print(f'{"tag":<20}{"n":>6}{"fin":>7}{"gait":>7}{"lat":>7}{"off50":>7}{"path":>7}{"colEp":>7}{"still":>7}')
rows = {}
for t in TAGS:
    m = load(t)
    if m is None:
        continue
    b = basic(m); rows[t] = (m, b)
    print(f'{t:<20}{b["n"]:>6}{b["fin"]:>7.3f}{b["gait"]:>7.3f}'
          f'{b.get("lat", float("nan")):>7.3f}{b.get("off50", float("nan")):>7.3f}'
          f'{b["path"]:>7.2f}{b["colEp"]:>7.3f}{b["still"]:>7.3f}')
for t, (m, _) in rows.items():
    c = vcurve(m)
    if c:
        print(f'\n[{t}] 속도 곡선  기울기 {c["slope"]:+.3f}  ' +
              " ".join(f"{x:.3f}->{y:.3f}" for x, y in zip(c["x"], c["y"])))
    tm = timing(m)
    if tm:
        if "beta" in tm:
            print(f'[{t}] 타이밍  beta {tm["beta"]:+.3f}  통과율 {tm["pass_rate"]:.2f} '
                  f'(n={tm["n"]})  encd {tm["encd"]:.3f}  latw {tm["latw"]:.3f}')
            print("        " + "  ".join(f"dt{x:.1f}->{y:+.3f}" for x, y in tm["curve"]))
        else:
            print(f'[{t}] 타이밍  {tm.get("note")} (n={tm["n"]})')


def obey_mode(m, A=2):
    """명령을 **어떻게** 따르는지 분해한다.

    보행이 한 가지로 고정돼 있으면 느린 명령을 따르는 길은 멈췄다 가기뿐이다.
    그러면 gait 은 그대로인데 정지 비율이 오른다. 어느 쪽인지 갈라야
    "속도를 조절한다" 와 "자주 멈춘다" 를 혼동하지 않는다.
    """
    if m.shape[1] < 31 or int(m[0, 25]) == 0:
        return None
    a = m[:, 0].astype(int) % A
    dtc = np.round(m[:, 19], 2)
    steps = np.maximum(m[:, 30], 1)
    move = np.maximum(steps - m[:, 4], 1)
    gait = m[:, 5] / (move / 30.0)
    stillf = m[:, 4] / steps
    arcrate = m[:, 24] / (steps / 30.0)          # 호길이 진척률
    sel = a == 1                                  # 감속 명령을 받는 쪽
    lv = sorted(set(dtc[sel]))
    if len(lv) < 2:
        return None
    out = []
    for L in lv:
        k = sel & (np.abs(dtc - L) < 1e-6)
        if k.sum() < 20:
            continue
        out.append((L, med(gait[k]), med(stillf[k]), med(arcrate[k])))
    return out


for t, (m, _) in rows.items():
    o = obey_mode(m)
    if o:
        print(f'\n[{t}] 명령을 어떻게 따르나 (a1, 감속 명령 받는 쪽)')
        print(f'    {"dt":>5}{"보행속도":>10}{"정지비율":>10}{"호길이진척":>11}')
        for L, g, s, r in o:
            print(f'    {L:>5.1f}{g:>10.3f}{s:>10.3f}{r:>11.3f}')
        if len(o) > 1:
            dg = o[-1][1] - o[0][1]; ds = o[-1][2] - o[0][2]
            print(f'    dt 0->{o[-1][0]:.1f} 변화:  보행 {dg:+.3f} m/s   정지비율 {ds:+.3f}')
            print(f'    -> {"보행을 늦춘다" if abs(dg) > 0.08 else "보행은 그대로, 더 자주 멈춘다"}')
