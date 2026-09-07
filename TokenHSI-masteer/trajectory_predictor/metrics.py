"""Open-loop metrics shared by validation and the standalone evaluator."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Union

import torch

from tokenhsi.utils import steer_path as sp

from .geometry import resample_coarse
from .oracle import timed_clearance
from .schema import PlannerState, state_from_batch


@torch.no_grad()
def batch_metrics(model, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    state = state_from_batch(batch).to(device)
    target = batch["coarse_path"].to(device)
    speed_target = batch["speed_class"].to(device)
    output = model(state)
    coarse = output["coarse_world"]
    speed = output["speed_logits"].argmax(dim=-1)
    dense, end_s = resample_coarse(coarse, with_end=True)
    b = coarse.shape[0]
    flat = dense.reshape(b * 2, sp.V, 2)
    box = state.box_xyz[..., :2].reshape(-1, 2)
    s_box, _, _ = sp.project(box, flat)
    stats = sp.path_stats(flat, s_box)
    feasible = (stats["turn_1.5m_deg"] <= 35.0 + 1e-4) & (end_s.reshape(-1) < (sp.V - 1) * sp.DS)
    clearance, _, _, _ = timed_clearance(
        dense[:, 0], dense[:, 1], end_s[:, 0], end_s[:, 1], speed[:, 0], speed[:, 1]
    )
    endpoint = torch.maximum(
        (coarse[:, :, 0] - state.root_xy).norm(dim=-1).amax(dim=1),
        torch.maximum(
            (coarse[:, :, 16] - state.box_xyz[..., :2]).norm(dim=-1).amax(dim=1),
            (coarse[:, :, 32] - state.goal_xy).norm(dim=-1).amax(dim=1),
        ),
    )
    ade = (coarse - target).norm(dim=-1).mean(dim=(1, 2))
    speed_acc = (speed == speed_target).float().mean(dim=(1, 2))
    return {
        "ade": ade,
        "endpoint_error": endpoint,
        "speed_accuracy": speed_acc,
        "path_feasible": feasible.reshape(b, 2).all(dim=1).float(),
        "clearance_ok": (clearance >= 1.0).float(),
        "min_clearance": clearance,
        "scenario": batch["scenario"].to(device),
    }


@torch.no_grad()
def evaluate_model(model, loader: Iterable, device: Union[str, torch.device]) -> Dict[str, float]:
    device = torch.device(device)
    values = defaultdict(list)
    model.eval()
    for batch in loader:
        result = batch_metrics(model, batch, device)
        for key, value in result.items():
            values[key].append(value.detach().cpu())
    merged = {key: torch.cat(parts) for key, parts in values.items()}
    metrics = {
        "ade": merged["ade"].mean().item(),
        "endpoint_error_max": merged["endpoint_error"].max().item(),
        "speed_accuracy": merged["speed_accuracy"].mean().item(),
        "path_feasibility": merged["path_feasible"].mean().item(),
        "clearance_rate": merged["clearance_ok"].mean().item(),
        "min_clearance_mean": merged["min_clearance"].mean().item(),
    }
    for value, name in ((0, "free"), (1, "cross")):
        mask = merged["scenario"] == value
        if mask.any():
            metrics[f"{name}_ade"] = merged["ade"][mask].mean().item()
            metrics[f"{name}_clearance_rate"] = merged["clearance_ok"][mask].mean().item()
    return metrics
