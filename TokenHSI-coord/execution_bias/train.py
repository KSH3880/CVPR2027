"""Offline supervised training for the deterministic execution-bias MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_bias_checkpoint
from .data import ExecutionBiasDataset
from .loss import execution_bias_loss
from .model import ExecutionBiasConfig, ExecutionBiasMLP


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    absolute_sum = 0.0
    valid_points = 0
    for plan, target, valid in loader:
        plan, target, valid = plan.to(device), target.to(device), valid.to(device)
        error = (model(plan) - target).norm(dim=-1)
        absolute_sum += float((error * valid).sum().item())
        valid_points += int(valid.sum().item())
    return absolute_sum / max(valid_points, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")

    train_set = ExecutionBiasDataset(args.train)
    validation_set = ExecutionBiasDataset(args.validation or args.train)
    agents, points = train_set.plan.shape[1:3]
    if validation_set.plan.shape[1:3] != (agents, points):
        raise ValueError("train and validation trajectory contracts differ")
    model = ExecutionBiasMLP(
        ExecutionBiasConfig(agents=agents, points=points)
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for plan, target, valid in train_loader:
            plan, target, valid = plan.to(args.device), target.to(args.device), valid.to(args.device)
            losses = execution_bias_loss(model(plan), target, plan, valid)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            running += float(losses["total"].detach().item())
            batches += 1
        validation_ade = evaluate(model, validation_loader, args.device)
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "validation_ade_m": validation_ade,
        }
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if validation_ade < best:
            best = validation_ade
            save_bias_checkpoint(args.output, model, step=epoch, metrics=metrics)


if __name__ == "__main__":
    main()
