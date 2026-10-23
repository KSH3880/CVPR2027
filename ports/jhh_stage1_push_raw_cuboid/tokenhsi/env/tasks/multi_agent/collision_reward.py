"""CPU/GPU agent-only distance and CPA risks, plus checkpoint reward contracts."""

import math
import torch


@torch.jit.script
def compute_agent_cpa_collision_penalty(root_pos, root_vel, min_dist, discount, dt):
    # type: (Tensor, Tensor, float, float, float) -> Tensor
    """MD's closing * predicted-distance-risk * per-control-step urgency, XY only."""
    p = root_pos[..., :2].unsqueeze(2) - root_pos[..., :2].unsqueeze(1)
    v = root_vel[..., :2].unsqueeze(2) - root_vel[..., :2].unsqueeze(1)
    dot = (p * v).sum(-1)
    speed_sq = v.square().sum(-1)
    denominator = (p.norm(p=2, dim=-1) * v.norm(p=2, dim=-1)).clamp_min(1e-06)
    closing = (-dot / denominator).clamp(0.0, 1.0)
    closest_time = (-dot / speed_sq.clamp_min(1e-06)).clamp_min(0.0)
    closest_distance = (p + v * closest_time.unsqueeze(-1)).norm(p=2, dim=-1)
    risk = ((min_dist - closest_distance) / min_dist).clamp(0.0, 1.0)
    urgency = torch.pow(torch.full_like(closest_time, discount), closest_time / dt)
    violation = closing * risk * urgency
    eye = torch.eye(
        root_pos.shape[1], device=root_pos.device, dtype=torch.bool
    ).unsqueeze(0)
    violation = torch.where(eye, torch.zeros_like(violation), violation)
    return violation.max(dim=-1)[0]


def agent_collision_config(env):
    """Resolve and validate one canonical signature for the independently paid term."""
    mode = env["agentCollisionMode"]
    if mode != "cpa":
        raise ValueError("Stage1 requires agentCollisionMode=cpa")
    enabled = env.get("agentCollisionPenalty", True)
    if type(enabled) is not bool:
        raise ValueError("agentCollisionPenalty must be boolean")
    result = dict(mode=mode, enabled=enabled)
    for key, name, default in (
        ("agentCollisionCoeff", "coefficient", 0.5),
        ("agentCollisionDist", "distance", 0.7),
    ):
        value = env.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not math.isfinite(value))
            or (value < 0)
            or (name == "distance" and value == 0)
        ):
            raise ValueError("Invalid " + key)
        result[name] = float(value)
    discount = env.get("agentCollisionTTCDiscount", 0.99)
    if (
        isinstance(discount, bool)
        or not isinstance(discount, (int, float))
        or (not math.isfinite(discount))
        or (not 0 < discount <= 1)
    ):
        raise ValueError("agentCollisionTTCDiscount must be in (0, 1]")
    result["ttc_discount"] = float(discount)
    return result


def check_agent_collision_checkpoint(weights, env):
    expected = agent_collision_config(env)
    saved = weights.get("agent_collision_config")
    if saved is None:
        old_env = weights.get("relation_experiment_config", {}).get("env")
        if old_env is not None:
            saved = agent_collision_config(old_env)
    if saved != expected:
        raise ValueError(
            "Checkpoint agent collision config differs; use matching Stage1 settings"
        )
