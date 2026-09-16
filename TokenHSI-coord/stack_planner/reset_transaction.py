"""Single-commit Isaac Gym reset for the carry-only stack environments."""

from __future__ import annotations

import torch
from isaacgym import gymtorch


class _DeferredResetGym:
    """Forward Gym calls except state setters/refreshes owned by the reset."""

    _DEFERRED = {
        "set_actor_root_state_tensor_indexed",
        "set_dof_state_tensor_indexed",
        "refresh_dof_state_tensor",
        "refresh_actor_root_state_tensor",
        "refresh_rigid_body_state_tensor",
        "refresh_force_sensor_tensor",
        "refresh_dof_force_tensor",
        "refresh_net_contact_force_tensor",
    }

    def __init__(self, gym):
        self._gym = gym

    def __getattr__(self, name):
        if name in self._DEFERRED:
            return lambda *args, **kwargs: None
        return getattr(self._gym, name)


class CarryOnlySingleCommitReset:
    """Coalesce inherited reset setters into one root and one DOF commit.

    The inherited multi-task/F22/sequential stack reset calls the indexed root
    setter several times before the next simulate call.  That can corrupt the
    Isaac Gym GPU pipeline.  Planner environments are carry-only and do not use
    the stack curriculum, so their complete reset can be staged in the wrapped
    tensors and committed once at the end.
    """

    def _reset_envs(self, env_ids):
        if len(env_ids) == 0 or not hasattr(self, "_root_states"):
            return super()._reset_envs(env_ids)
        if getattr(self, "stack_curriculum", False):
            raise RuntimeError(
                "single-commit planner reset requires STACK_CURRICULUM=0"
            )
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        real_gym = self.gym
        self.gym = _DeferredResetGym(real_gym)
        try:
            # Every inherited tensor mutation and bookkeeping operation still
            # runs. Only Gym state submission and premature refresh are held.
            super()._reset_envs(env_ids)
        finally:
            self.gym = real_gym

        actors_per_env = self._root_states.shape[0] // self.num_envs
        local_actor = torch.arange(
            actors_per_env, device=self.device, dtype=torch.int32
        )
        root_actor_ids = (
            env_ids.to(torch.int32)[:, None] * actors_per_env
            + local_actor[None, :]
        ).reshape(-1).contiguous()
        humanoid_actor_ids = self._humanoid_actor_ids_per_env[
            env_ids
        ].reshape(-1).contiguous()

        # One call per setter between simulation steps is the supported Isaac
        # Gym tensor API pattern. Committing every actor also includes objects,
        # platforms, and viewer markers modified by inherited reset hooks.
        real_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(root_actor_ids),
            len(root_actor_ids),
        )
        real_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(humanoid_actor_ids),
            len(humanoid_actor_ids),
        )
        self._refresh_sim_tensors()

        # Parent observation/AMP initialization ran while refreshes were held.
        # Rebuild them once from the committed simulator tensors.
        self._reset_progress_ref(env_ids)
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)


__all__ = ["CarryOnlySingleCommitReset"]
