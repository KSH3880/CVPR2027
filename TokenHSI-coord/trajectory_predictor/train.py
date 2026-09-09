"""CLI: supervised task-specific pretraining on oracle-valid scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .checkpoint import file_sha256, load_checkpoint, save_checkpoint
from .dataset import TrajectoryDataset
from .geometry import state_to_tokens
from .losses import compute_loss
from .metrics import evaluate_model
from .model import JointTrajectoryPredictor, ModelConfig
from .schema import state_from_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("../runs/trajectory_predictor/checkpoints"))
    parser.add_argument("--name", default="joint_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_parts = [TrajectoryDataset(path, augment=True, valid_only=True) for path in args.train]
    train_set = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_set = TrajectoryDataset(args.val, augment=False, valid_only=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.init_checkpoint is None:
        model = JointTrajectoryPredictor(ModelConfig()).to(device)
        token_parts = []
        for dataset in train_parts:
            tokens, _ = state_to_tokens(dataset.state.index(dataset.indices))
            token_parts.append(tokens)
        model.fit_normalizer(torch.cat(token_parts).to(device))
    else:
        model, _ = load_checkpoint(args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dataset_hash = hashlib.sha256(
        "".join(file_sha256(path) for path in args.train).encode("ascii")
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.name}_metrics.jsonl"
    best_key = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {key: 0.0 for key in ("total", "waypoint", "speed", "smoothness", "collision")}
        seen = 0
        for batch in train_loader:
            state = state_from_batch(batch).to(device)
            target_path = batch["coarse_path"].to(device)
            target_speed = batch["speed_class"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(state), state, target_path, target_speed)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = state.batch_size
            seen += n
            for key in sums:
                sums[key] += loss[key].detach().item() * n

        metrics = evaluate_model(model, val_loader, device)
        train_metrics = {f"train_{key}": value / max(seen, 1) for key, value in sums.items()}
        report = {"epoch": epoch, **train_metrics, **metrics}
        print(json.dumps(report, sort_keys=True), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, sort_keys=True) + "\n")
        save_checkpoint(
            args.output_dir / f"{args.name}_last.pth", model, dataset_hash,
            epoch=epoch, metrics=metrics, optimizer=optimizer,
        )
        key = (metrics["clearance_rate"], -metrics["ade"])
        if best_key is None or key > best_key:
            best_key = key
            save_checkpoint(
                args.output_dir / f"{args.name}_best.pth", model, dataset_hash,
                epoch=epoch, metrics=metrics,
            )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            snapshot = args.output_dir / f"{args.name}_e{epoch:04d}.pth"
            save_checkpoint(snapshot, model, dataset_hash, epoch=epoch, metrics=metrics)
            print(f"saved numbered checkpoint: {snapshot}", flush=True)


if __name__ == "__main__":
    main()
