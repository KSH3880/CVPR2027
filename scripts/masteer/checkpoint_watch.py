#!/usr/bin/env python3
"""실행 중인 masteer의 덮어쓰기 체크포인트를 epoch 단위로 보존한다.

이미 시작한 rl_games 프로세스는 save_intermediate 설정을 다시 읽지 않는다. 이 watcher는
Humanoid.pth가 갱신될 때 안정된 사본을 만든 뒤 내부 epoch를 확인하고, 지정한 간격의
체크포인트만 같은 nn 디렉터리에 Humanoid_<epoch>.pth로 보존한다.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def process_alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def newest_checkpoint(run_root: Path):
    paths = list(run_root.glob("*/nn/Humanoid.pth"))
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def stable_snapshot(source: Path):
    """동시에 쓰는 파일의 부분 사본을 채택하지 않는다."""
    before = source.stat()
    temp = source.with_name(f".{source.name}.watch-{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temp)
        after = source.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            return None, None
        state = torch.load(temp, map_location="cpu")
        return temp, int(state["epoch"])
    except (EOFError, OSError, RuntimeError, KeyError, ValueError):
        return None, None
    finally:
        if temp.exists():
            temp.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--max-epoch", type=int, default=9000)
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()
    if args.every <= 0 or args.max_epoch <= 0 or args.poll <= 0:
        parser.error("every, max-epoch, poll은 양수여야 합니다")

    run_root = ROOT / "TokenHSI-masteer" / "output" / "masteer" / args.tag
    pid_file = ROOT / "runs" / "queue" / "logs" / f"{args.tag}.pid"
    seen = None
    print(
        f"[ckpt-watch] tag={args.tag} every={args.every} max={args.max_epoch}",
        flush=True,
    )

    while True:
        source = newest_checkpoint(run_root)
        if source is not None:
            try:
                signature = (source.stat().st_mtime_ns, source.stat().st_size)
            except FileNotFoundError:
                signature = None
            if signature is not None and signature != seen:
                temp, epoch = stable_snapshot(source)
                if temp is not None:
                    seen = signature
                    if epoch % args.every == 0:
                        target = source.with_name(f"Humanoid_{epoch:08d}.pth")
                        if target.exists():
                            print(f"[ckpt-watch] already exists epoch={epoch}: {target}", flush=True)
                        else:
                            # stable_snapshot의 임시 파일은 finally에서 지워지므로 검증된
                            # 원본을 한 번 더 복사하고 원자적으로 이름을 바꾼다.
                            final_temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                            shutil.copyfile(source, final_temp)
                            check = torch.load(final_temp, map_location="cpu")
                            if int(check["epoch"]) != epoch:
                                final_temp.unlink(missing_ok=True)
                                time.sleep(args.poll)
                                continue
                            os.replace(final_temp, target)
                            print(f"[ckpt-watch] saved epoch={epoch}: {target}", flush=True)
                        if epoch >= args.max_epoch:
                            print("[ckpt-watch] target epoch reached", flush=True)
                            return 0

        if not process_alive(pid_file):
            print("[ckpt-watch] training process ended", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
