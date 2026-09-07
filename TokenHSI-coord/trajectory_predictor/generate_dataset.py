"""CLI: generate deterministic V1 oracle datasets on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DEFAULT_SPLITS, SPLIT_SEEDS, generate_split, save_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(DEFAULT_SPLITS), required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    count = DEFAULT_SPLITS[args.split] if args.count is None else args.count
    seed = SPLIT_SEEDS[args.split] if args.seed is None else args.seed
    output = args.output or Path("../runs/trajectory_predictor/data") / f"joint_v1_{args.split}.pt"
    payload = generate_split(count, seed, args.batch_size)
    save_split(output, payload)
    valid = payload["oracle_valid"].float().mean().item()
    print(f"saved {count} {args.split} scenes to {output} (oracle_valid={valid:.4f})", flush=True)


if __name__ == "__main__":
    main()
