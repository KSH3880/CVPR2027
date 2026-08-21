#!/usr/bin/env python3
"""고정 카메라 영상에서 **파란 에이전트의 실제 이동 픽셀**을 낸다.

픽셀 변화량 비율은 못 쓴다 -- 위에서 보면 사람이 화면의 1% 미만이라
굳은 영상 0.24 대 움직인 영상 0.34 로 거의 안 갈린다 (실측).
agent 1 의 몸 색(0.28, 0.60, 0.93)은 바닥·경로·창 어느 것과도 겹치지 않아
무게중심을 프레임마다 찍으면 이동 거리가 바로 나온다.
"""
import subprocess, sys, os, numpy as np
from PIL import Image

mp4 = sys.argv[1]
b = os.path.basename(mp4)[:-4]
os.makedirs("/tmp/frames", exist_ok=True)
FR = [60, 210, 360, 510, 660]          # 2s ~ 22s, 한 에피소드를 덮는다
sel = "+".join(f"eq(n\\,{f})" for f in FR)
subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", mp4,
                "-vf", f"select='{sel}',scale=680:-1", "-vsync", "0",
                f"/tmp/frames/mv_{b}_%d.png"], check=False)

cents = []
for i in range(1, len(FR) + 1):
    p = f"/tmp/frames/mv_{b}_{i}.png"
    if not os.path.exists(p):
        continue
    a = np.asarray(Image.open(p).convert("RGB"), dtype=int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    # 파란 사람: B 가 높고 G 는 중간, R 은 낮다. 시안(창 띠)은 G 가 매우 높아 걸러진다.
    m = (B > 190) & (G > 110) & (G < 200) & (R < 120)
    if m.sum() < 12:
        continue
    ys, xs = np.nonzero(m)
    cents.append((xs.mean(), ys.mean()))

if len(cents) < 2:
    print("0.0"); sys.exit()
c = np.array(cents)
print(f"{float(np.abs(c - c[0]).sum(axis=1).max()):.1f}")
