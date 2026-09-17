"""Same-state counterfactual rollout snapshots for the stack planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


_DERIVED_TENSOR_NAMES = (
    "rigid_body", "contact_force", "force_sensor", "dof_force",
)


@dataclass
class TaskBranchSnapshot:
    """Mutable per-environment task tensors plus physical simulator state."""

    tensors: Dict[str, torch.Tensor]
    factors: Dict[str, int]
    num_envs: int

    @classmethod
    @torch.no_grad()
    def capture(cls, task):
        num_envs = int(task.num_envs)
        tensors = {}
        factors = {}
        for name, value in vars(task).items():
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            if any(part in name for part in _DERIVED_TENSOR_NAMES):
                continue
            if value.shape[0] % num_envs:
                continue
            factor = value.shape[0] // num_envs
            # Large dataset/model tables are not per-environment state.  All
            # mutable task buffers in this environment have a modest leading
            # factor (agents, actors, DOFs or path rows).
            if factor < 1 or factor > 256:
                continue
            tensors[name] = value.detach().clone()
            factors[name] = factor
        for required in ("_root_states", "_dof_state"):
            if required not in tensors:
                raise RuntimeError(f"branch snapshot missing {required}")
        return cls(tensors=tensors, factors=factors, num_envs=num_envs)

    def clone(self):
        return TaskBranchSnapshot(
            {name: value.clone() for name, value in self.tensors.items()},
            dict(self.factors), self.num_envs,
        )

    @torch.no_grad()
    def update_where(self, source: "TaskBranchSnapshot", env_mask):
        if self.num_envs != source.num_envs or self.factors != source.factors:
            raise ValueError("branch snapshot layouts do not match")
        mask = torch.as_tensor(env_mask, dtype=torch.bool,
                               device=next(iter(self.tensors.values())).device)
        if mask.shape != (self.num_envs,):
            raise ValueError("env_mask must be [num_envs]")
        for name, target in self.tensors.items():
            factor = self.factors[name]
            target_view = target.reshape(self.num_envs, factor, *target.shape[1:])
            source_view = source.tensors[name].reshape(
                self.num_envs, factor, *target.shape[1:]
            )
            target_view[mask] = source_view[mask]

    @torch.no_grad()
    def restore(self, task):
        # Keep snapshot bookkeeping unit-testable without importing Isaac Gym;
        # the trainer itself has already imported gymapi before torch.
        from isaacgym import gymtorch

        if int(task.num_envs) != self.num_envs:
            raise ValueError("snapshot num_envs mismatch")
        for name, saved in self.tensors.items():
            current = getattr(task, name, None)
            if not torch.is_tensor(current) or current.shape != saved.shape:
                raise RuntimeError(f"branch tensor changed: {name}")
            current.copy_(saved)
        task.gym.set_actor_root_state_tensor(
            task.sim, gymtorch.unwrap_tensor(task._root_states),
        )
        task.gym.set_dof_state_tensor(
            task.sim, gymtorch.unwrap_tensor(task._dof_state),
        )
        task._refresh_sim_tensors()


__all__ = ["TaskBranchSnapshot"]
