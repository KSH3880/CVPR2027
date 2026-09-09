"""CLI wrapper for the pure 2D checkpoint video renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .visualize import render_checkpoint_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()
    path = render_checkpoint_video(
        args.checkpoint, args.data, args.output,
        scenes=args.scenes, fps=args.fps, max_frames=args.max_frames,
    )
    print(f"saved trajectory video: {path}", flush=True)


if __name__ == "__main__":
    main()
