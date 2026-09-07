"""Render each atomic numbered planner checkpoint in a separate process."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict

from .visualize import render_checkpoint_video


def _numbered_checkpoints(directory: Path, name: str) -> Dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(name)}_e(\d+)\.pth$")
    result = {}
    for path in directory.glob(f"{name}_e*.pth"):
        match = pattern.match(path.name)
        if match is not None:
            result[int(match.group(1))] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="joint_v1")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--final-epoch", type=int, default=None)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.final_epoch is not None and args.final_epoch <= 0:
        parser.error("--final-epoch must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"watching {args.checkpoint_dir}/{args.name}_e*.pth "
        f"(validation={args.data})",
        flush=True,
    )
    failures: Dict[int, str] = {}
    while True:
        checkpoints = _numbered_checkpoints(args.checkpoint_dir, args.name)
        for epoch, checkpoint in sorted(checkpoints.items()):
            output = args.output_dir / f"{args.name}_e{epoch:04d}.mp4"
            if output.is_file() and output.stat().st_size > 0:
                failures.pop(epoch, None)
                continue
            try:
                render_checkpoint_video(
                    checkpoint,
                    args.data,
                    output,
                    scenes=args.scenes,
                    fps=args.fps,
                    max_frames=args.max_frames,
                )
            except Exception as error:  # Keep watching later checkpoints.
                message = f"{type(error).__name__}: {error}"
                if failures.get(epoch) != message:
                    print(f"video failed for epoch {epoch}: {message}", flush=True)
                failures[epoch] = message
            else:
                failures.pop(epoch, None)
                print(f"saved trajectory video: {output}", flush=True)

        final_output = None
        if args.final_epoch is not None:
            final_output = args.output_dir / f"{args.name}_e{args.final_epoch:04d}.mp4"
        if args.once or (final_output is not None and final_output.is_file() and final_output.stat().st_size > 0):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
