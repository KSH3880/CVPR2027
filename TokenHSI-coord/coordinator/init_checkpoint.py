"""Create a random C1 checkpoint for import/bridge smoke tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .model import JointCoordinator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = JointCoordinator()
    save_checkpoint(args.output, model, metrics={"random_init": 1.0})
    print(f"saved random-init smoke checkpoint: {args.output}")


if __name__ == "__main__":
    main()
