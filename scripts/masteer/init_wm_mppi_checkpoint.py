#!/usr/bin/env python3
"""Create an untrained coordinator_v2 initialization checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TokenHSI-masteer"))

from coordinator_v2 import PlannerConfig, WorldModelPlanner, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    config = PlannerConfig(mppi_candidates=args.candidates, mppi_iterations=args.iterations)
    save_checkpoint(args.output, WorldModelPlanner(config), extras={"trained": False})
    print(f"wrote untrained coordinator_v2 checkpoint: {args.output}")


if __name__ == "__main__":
    main()
