#!/usr/bin/env python3
"""Per-command-band speed tracking, which the eight-column result row cannot show.

    python scripts/steer/speed_report.py f35_m4 f35_m8 ...

success and app_xt are both blind to speed: a policy can follow the path exactly
at the wrong pace and score well on both. Needs a run logged with STEER_MRAND>0,
which records cmdv (commanded) and actv (actual) per frame.

Stopped frames are dropped. Grasping, putting down, and standing at a delivered
target all sit near zero actual speed regardless of what was commanded, and they
are a third of the episode -- leaving them in drags every band's median toward
zero and hides the tracking entirely.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/hwanhee/CVPR2027")
BANDS = [(0.30, 0.50), (0.50, 0.90), (0.90, 1.30), (1.30, 1.70)]


def report(tag):
    p = ROOT / "runs/steer" / f"{tag}.npz"
    if not p.exists():
        print(f"{tag}: npz 없음"); return
    d = np.load(p)
    if "cmdv" not in d:
        print(f"{tag}: cmdv 없음 (STEER_MRAND 로 학습/평가한 런이 아니다)"); return
    c, a = d["cmdv"].ravel(), d["actv"].ravel()
    keep = (c > 0.01) & (a > 0.05)
    c, a = c[keep], a[keep]
    print(f"\n=== {tag} ===  {len(c):,} 프레임 (정지 제외)")
    print(f"{'명령':>12} {'n':>8} {'실제 중앙':>10} {'p25':>7} {'p75':>7} {'오차':>7}")
    for lo, hi in BANDS:
        s = (c >= lo) & (c < hi)
        if s.sum() < 200:
            continue
        err = np.median(np.abs(a[s] - c[s]))
        print(f"{lo:5.2f}~{hi:<5.2f} {s.sum():8,} {np.median(a[s]):10.2f} "
              f"{np.percentile(a[s], 25):7.2f} {np.percentile(a[s], 75):7.2f} {err:7.3f}")
    r = np.corrcoef(c, a)[0, 1]
    print(f"{'전체':>12} {len(c):8,} {'':10} {'':7} {'':7} "
          f"{np.median(np.abs(a - c)):7.3f}   상관 {r:.3f}")


if __name__ == "__main__":
    for t in sys.argv[1:] or ["f35_m4"]:
        report(t)
