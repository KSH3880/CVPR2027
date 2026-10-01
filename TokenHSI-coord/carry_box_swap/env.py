"""A=2 original stage1 Carry with externally assigned box observations."""

import os
import torch
import numpy as np

from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
from carry_box_swap.assignment import assigned_inputs, assignment_rows


class HumanoidCarryBoxSwapProbe(HumanoidTrajSitCarryClimb):
    def __init__(self, cfg, *args, **kwargs):
        self.goals_follow_box = bool(int(os.environ.get("CARRY_BOX_SWAP_GOALS_FOLLOW_BOX", "1")))
        super().__init__(cfg, *args, **kwargs)
        if self.num_agents != 2:
            raise ValueError("box-swap probe requires two agents")
        self._enable_IET = False
        self._ensure_assignment()

    def _ensure_assignment(self):
        if not hasattr(self, "box_assignment"):
            self.box_assignment = torch.arange(2, device=self.device)[None].expand(self.num_envs, -1).clone()

    def _reset_envs(self, env_ids):
        self._ensure_assignment()
        self.box_assignment[env_ids] = torch.tensor([0, 1], device=self.device)
        super()._reset_envs(env_ids)

    def set_box_assignment(self, env_ids, assignment):
        assignment_rows(assignment)
        if assignment.shape[0] != len(env_ids):
            raise ValueError("assignment batch must match env_ids")
        self._ensure_assignment()
        self.box_assignment[env_ids] = assignment

    def _compute_task_obs(self, env_ids=None):
        self._ensure_assignment()
        with assigned_inputs(self, self.goals_follow_box):
            return super()._compute_task_obs(env_ids)

    def _compute_reward(self, actions):
        self._ensure_assignment()
        with assigned_inputs(self, self.goals_follow_box):
            return super()._compute_reward(actions)

    def _draw_task(self):
        super()._draw_task()
        root = self.humanoid_rows(self._humanoid_root_states).reshape(self.num_envs, 2, -1)
        boxes = self.humanoid_rows(self._box_states)[assignment_rows(self.box_assignment)].reshape(self.num_envs, 2, -1)
        lines = torch.cat((root[..., :3], boxes[..., :3]), -1).cpu().numpy()
        colors = np.asarray([[0., 1., 1.], [1., .5, 0.]], dtype=np.float32)
        for env_id, env in enumerate(self.envs):
            self.gym.add_lines(self.viewer, env, 2, lines[env_id], colors)
