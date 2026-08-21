#!/usr/bin/env python3
"""Pulls training curves out of the tensorboard summaries.

    python scripts/steer/curves.py [name ...]

Prints, per run, the trend of the signals that actually say whether learning is
working: episode return, the AMP discriminator's accuracy on agent vs demo
motion, and the policy/value losses.
"""

import glob
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

ROOT = Path("/home/hwanhee/CVPR2027/TokenHSI-steer/output/steer")
WANT = ["rewards0/iter", "info/disc_reward_mean", "info/disc_agent_acc",
        "info/disc_demo_acc", "losses/a_loss", "losses/c_loss",
        "losses/entropy", "info/kl"]


def series(acc, tag):
    if tag not in acc.Tags()["scalars"]:
        return None
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def trend(y, frac=0.15):
    """Mean of the first and last slice, so a curve reads as start -> now."""
    n = max(int(len(y) * frac), 1)
    return float(y[:n].mean()), float(y[-n:].mean())


def report(name):
    files = sorted(glob.glob(str(ROOT / name / "*" / "summaries" / "events*")))
    if not files:
        print(f"{name}: 요약 파일 없음")
        return
    acc = ea.EventAccumulator(files[-1], size_guidance={ea.SCALARS: 0})
    acc.Reload()

    r = series(acc, "rewards0/iter")
    if r is None:
        print(f"{name}: 보상 기록 없음")
        return
    steps, rew = r
    a, b = trend(rew)
    peak = float(rew.max())
    print(f"\n{name}   epoch {int(steps[-1])}")
    print(f"  보상        {a:.3f} → {b:.3f}   (최고 {peak:.3f}, "
          f"{'상승' if b > a * 1.02 else '정체' if b > a * 0.98 else '하락'})")
    for tag, label in (("info/disc_reward_mean", "AMP 보상  "),
                       ("info/disc_agent_acc", "판별자 agent"),
                       ("info/disc_demo_acc", "판별자 demo "),
                       ("losses/a_loss", "policy loss"),
                       ("losses/c_loss", "value loss "),
                       ("info/kl", "KL        ")):
        s = series(acc, tag)
        if s is None:
            continue
        a2, b2 = trend(s[1])
        print(f"  {label}  {a2:.4f} → {b2:.4f}")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    for n in names:
        report(n)
