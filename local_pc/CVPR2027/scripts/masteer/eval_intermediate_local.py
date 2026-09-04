#!/usr/bin/env python3
"""masteer 중간 체크포인트를 순차 평가한다.

기본 모드는 학습 종료 후 평가하고, ``--live`` 는 학습 중 새 체크포인트가
생길 때마다 평가한다. 체크포인트 파일을 쓰는 도중 읽지 않도록 mtime 안정화
시간을 둔다.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def process_alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def checkpoint_epoch(path: Path):
    match = re.fullmatch(r"Humanoid_0*([0-9]+)\.pth", path.name)
    return int(match.group(1)) if match else None


def already_evaluated(label: str) -> bool:
    log = ROOT / "runs" / "queue" / "logs" / f"eval_{label}.log"
    metrics = ROOT / "runs" / "results" / "masteer" / f"eval_{label}.npy"
    return metrics.exists() and log.exists() and "MS_EVAL_SUMMARY " in log.read_text(errors="ignore")


def discover_checkpoints(run_root: Path, ready_age: float):
    now = time.time()
    paths = {}
    for path in run_root.glob("*/nn/Humanoid_[0-9]*.pth"):
        epoch = checkpoint_epoch(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if epoch is not None and stat.st_size > 0 and now - stat.st_mtime >= ready_age:
            paths[epoch] = path.resolve()
    return paths


def evaluate(args, epoch: int, checkpoint: Path):
    label = f"{args.tag}_e{epoch}"
    if already_evaluated(label):
        print(f"[eval-watch] already evaluated: {label}", flush=True)
        return True
    env = os.environ.copy()
    env["QUEUE_EVAL_SOURCE"] = args.tag
    env["QUEUE_EVAL_CKPT"] = str(checkpoint)
    print(f"[eval-watch] evaluating {label}: {checkpoint}", flush=True)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "masteer" / "eval_one.sh"),
            label,
            args.gpu,
            str(args.envs),
        ],
        cwd=str(ROOT),
        env=env,
    )
    if result.returncode != 0:
        print(f"[eval-watch] failed {label}: rc={result.returncode}", flush=True)
        return False
    print(f"[eval-watch] complete {label}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--start-epoch", type=int, default=1000)
    parser.add_argument("--max-epoch", type=int, default=9000)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--settle", type=float, default=30.0)
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--checkpoint-settle", type=float, default=10.0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="학습 종료를 기다리지 않고 새 체크포인트가 생길 때마다 평가",
    )
    args = parser.parse_args()
    if min(args.every, args.start_epoch, args.max_epoch, args.envs) <= 0:
        parser.error("epoch 간격과 env 수는 양수여야 합니다")
    if min(args.settle, args.poll, args.checkpoint_settle) < 0:
        parser.error("대기 시간은 음수일 수 없습니다")

    pid_file = ROOT / "runs" / "queue" / "logs" / f"{args.tag}.pid"
    run_root = ROOT / "TokenHSI-masteer" / "output" / "masteer" / args.tag
    print(
        f"[eval-watch] tag={args.tag} every={args.every} "
        f"range={args.start_epoch}..{args.max_epoch} envs={args.envs} "
        f"mode={'live' if args.live else 'after-training'}",
        flush=True,
    )
    expected = list(range(args.start_epoch, args.max_epoch + 1, args.every))
    failures = []
    attempted = set()

    if not args.live:
        while process_alive(pid_file):
            time.sleep(args.poll)
        # PhysX/PyTorch가 GPU context를 완전히 내릴 시간을 둔다.
        print(f"[eval-watch] training ended; settling GPU for {args.settle:.0f}s", flush=True)
        time.sleep(args.settle)

    while True:
        paths = discover_checkpoints(run_root, args.checkpoint_settle)
        for epoch in expected:
            checkpoint = paths.get(epoch)
            if checkpoint is None or epoch in attempted:
                continue
            attempted.add(epoch)
            if not evaluate(args, epoch, checkpoint):
                failures.append(epoch)

        alive = process_alive(pid_file)
        if not args.live or not alive:
            missing = [epoch for epoch in expected if epoch not in paths]
            if missing:
                print(f"[eval-watch] unavailable checkpoints: {missing}", flush=True)
            break
        time.sleep(args.poll)

    if failures:
        print(f"[eval-watch] failures={failures}", flush=True)
        return 1
    print("[eval-watch] all available checkpoints evaluated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
