"""Track whether each live training episode has had a collision proxy hit."""

from __future__ import annotations

import torch


class EpisodeCollisionTracker:
    def __init__(self, num_envs: int, device: torch.device):
        self.collided = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, collision: torch.Tensor, done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (collided, completed) for episodes ending on this step."""
        if collision.shape != self.collided.shape or done.shape != self.collided.shape:
            raise ValueError("episode collision masks must have shape [num_envs]")
        self.collided |= collision.bool()
        collided_done = (self.collided & done).sum()
        completed = done.sum()
        self.collided &= ~done
        return collided_done, completed
