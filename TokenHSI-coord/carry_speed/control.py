"""Simulator-independent speed commands, scenes and PPO returns."""

import torch

SPEED_CHOICES = (0.0, 0.375, 0.75, 1.125, 1.5)


def rate_limit_speed(current, desired, dt, up, down):
    delta = (desired - current).clamp(min=-down * dt, max=up * dt)
    return (current + delta).clamp(0.0, SPEED_CHOICES[-1])


def fixed_route(start, goal):
    unit = torch.linspace(0, 1, 33, device=start.device, dtype=start.dtype)
    return start[..., None, :] + unit[None, None, :, None] * (goal - start)[..., None, :]


def scene_geometry(root, center, distance, jitter, free):
    """Cross at the shared center; separated parallel routes are controls."""
    direction = torch.nn.functional.normalize(root - center[:, None], dim=-1)
    start = center[:, None] + direction * (distance[:, None] + jitter)[..., None]
    goal = center[:, None] - 4.0 * direction
    # The randomized geometric slot determines lane assignment, not agent ID.
    lane = torch.where(direction[..., 0].abs() > 0.5, -1.5, 1.5)
    parallel_start = center[:, None] + torch.stack((-distance[:, None].expand_as(lane), lane), -1)
    parallel_goal = center[:, None] + torch.stack((torch.full_like(lane, 4.0), lane), -1)
    return (torch.where(free[:, None, None], parallel_start, start),
            torch.where(free[:, None, None], parallel_goal, goal))


def controlled_mask(priority):
    return torch.arange(2, device=priority.device)[None] != priority[:, None]


def action_speeds(action, priority):
    speed = torch.tensor(SPEED_CHOICES, device=action.device)[action]
    return torch.where(controlled_mask(priority), speed, torch.full_like(speed, SPEED_CHOICES[-1]))


def rule_speeds(root, center, goal, priority):
    """A geometric reference controller; its physical safety is not guaranteed."""
    forward = torch.nn.functional.normalize(goal - root, dim=-1)
    remaining = ((center[:, None] - root) * forward).sum(-1)
    leader_remaining = remaining.gather(1, priority[:, None])
    # Wait outside a 1.6m conflict region until the leader is 1.6m beyond it.
    wait = (leader_remaining > -1.6) & (remaining > -1.6) & (remaining < 3.0)
    speed = torch.full_like(remaining, SPEED_CHOICES[-1])
    return torch.where(wait & controlled_mask(priority), torch.zeros_like(speed), speed)


def generalized_advantages(reward, done, value, next_value, gamma=0.99, lam=0.95):
    advantage = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for index in range(reward.shape[0] - 1, -1, -1):
        live = (~done[index]).to(reward.dtype)
        delta = reward[index] + gamma * next_value[index] * live - value[index]
        carry = delta + gamma * lam * live * carry
        advantage[index] = carry
    return advantage, advantage + value
