"""Differentiable adapter from planner candidates to predicted executed paths."""

from __future__ import annotations

from typing import Mapping

import torch

from .model import ExecutionBiasMLP


def predict_candidate_execution(
    path_world: torch.Tensor,
    speed: torch.Tensor,
    model: ExecutionBiasMLP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct [B,K,A,P] planner candidates with a frozen bias model.

    Returns ``(executed_path_world, error_world)`` with the same candidate
    dimensions as ``path_world``.  No detach is used, so a differentiable
    planner cost can optimize through the frozen model back into its plan.
    """
    if path_world.ndim != 5 or path_world.shape[-1] != 2:
        raise ValueError("path_world must be [B,K,A,P,2]")
    if speed.shape != path_world.shape[:-1]:
        raise ValueError("speed must be [B,K,A,P]")
    batch, candidates, agents, points, _ = path_world.shape
    if (agents, points) != (model.config.agents, model.config.points):
        raise ValueError(
            "planner/model contract mismatch: "
            f"planner A/P={(agents, points)} model A/P="
            f"{(model.config.agents, model.config.points)}"
        )
    plan = torch.cat((path_world, speed[..., None]), dim=-1).reshape(
        batch * candidates, agents, points, 3
    )
    error = model(plan).reshape(batch, candidates, agents, points, 2)
    return path_world + error, error


def attach_execution_prediction(
    output: Mapping[str, torch.Tensor],
    model: ExecutionBiasMLP,
) -> dict[str, torch.Tensor]:
    """Copy planner output and add deterministic execution prediction fields."""
    if "path_world" not in output or "speed" not in output:
        raise ValueError("planner output must contain path_world and speed")
    executed, error = predict_candidate_execution(
        output["path_world"], output["speed"], model
    )
    result = dict(output)
    result["execution_error_xy"] = error
    result["executed_path_world"] = executed
    return result
