"""장기런의 롤링 체크포인트가 목표 epoch 을 넘는 순간 스냅샷을 뜬다.

save_intermediate 가 꺼져 있어 rl_games 는 Humanoid.pth 를 100 epoch 마다
덮어쓰기만 한다. 9000 런에서 3000 시점을 평가하려면 지나가는 순간 잡아야 한다.

**임시 파일로 먼저 복사한 뒤 epoch 을 읽는다.** 원본을 읽고 나서 복사하면
그 사이에 학습이 덮어쓸 수 있다.
"""
import os, shutil, sys, time
import torch

TARGET = int(os.environ.get("SNAP_EPOCH", 3000))
TAGS = sys.argv[1:] or ["ms12_base4L_s0", "ms13_sep9L_s0", "ms13_sep20L_s0"]
ROOT = "/home/hwanhee/CVPR2027/TokenHSI-masteer/output/masteer"


def rolling(tag):
    import glob
    c = sorted(glob.glob(f"{ROOT}/{tag}/Humanoid_*/nn/Humanoid.pth"),
               key=os.path.getmtime)
    return c[-1] if c else None


done = set()
while len(done) < len(TAGS):
    for tag in TAGS:
        if tag in done:
            continue
        src = rolling(tag)
        if not src:
            continue
        tmp = src + ".snaptmp"
        try:
            shutil.copy2(src, tmp)
            ep = torch.load(tmp, map_location="cpu", weights_only=False).get("epoch", 0)
        except Exception as e:
            os.path.exists(tmp) and os.remove(tmp)
            print(f"[snap] {tag} 읽기 실패, 다음 주기에 재시도: {e}", flush=True)
            continue
        if ep >= TARGET:
            dst = os.path.join(os.path.dirname(src), f"Humanoid_ep{ep}.pth")
            os.replace(tmp, dst)
            done.add(tag)
            print(f"[snap] {tag} epoch={ep} -> {dst}", flush=True)
        else:
            os.remove(tmp)
    if len(done) < len(TAGS):
        time.sleep(600)
print(f"[snap] 완료: {sorted(done)}", flush=True)
