"""CLI: held-out Free/Cross acceptance metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .dataset import TrajectoryDataset
from .metrics import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint, args.device)
    dataset = TrajectoryDataset(args.data, augment=False, valid_only=True)
    metrics = evaluate_model(model, DataLoader(dataset, batch_size=args.batch_size), args.device)
    passed = (
        metrics["endpoint_error_max"] <= 0.01
        and metrics["speed_accuracy"] >= 0.95
        and metrics["path_feasibility"] >= 0.99
        and metrics["clearance_rate"] >= 0.99
    )
    print(json.dumps({**metrics, "accepted": passed}, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
