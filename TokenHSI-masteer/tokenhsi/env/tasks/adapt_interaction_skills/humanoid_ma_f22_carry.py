"""Run the single-agent f22 steering policy in the safe A=2 environment.

The policy contract stays exactly [self 223] + [dual steer 24] + [carry 42].
No teammate token or extra carry copy is added, so the f22 checkpoint can be
loaded without changing its network shape.
"""

import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_carry import (
    HumanoidMACarry, CARRY_LO, CARRY_HI, TEAMMATE_DIM,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import HumanoidMASteerCarry
from tokenhsi.utils import steer_path as sp


class HumanoidMAF22Carry(HumanoidMASteerCarry):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # f22 has no multi-task one-hot in AMP observations. The discriminator is
        # unused at test time, but strict loading still requires its 1290 input.
        cfg["env"]["enableTaskSpecificDisc"] = False
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)


    def steer_dim(self):
        return 4 * self.steer_k

    def get_task_obs_size(self):
        return self.steer_dim() + (CARRY_HI - CARRY_LO)

    def get_multi_task_info(self):
        if getattr(self, "_use_base_info", False):
            return super().get_multi_task_info()
        each = [self.steer_dim(), CARRY_HI - CARRY_LO, CARRY_HI - CARRY_LO]
        names = ["new_extra", "new_carry", "old_carry"]
        n = len(each)
        mask = torch.zeros(n, sum(each), dtype=torch.bool, device=self.device)
        index = torch.cumsum(torch.tensor([0] + each), dim=0).to(self.device)
        for i in range(n):
            mask[i, index[i]:index[i + 1]] = True
        return {
            "onehot_size": n,
            "tota_subtask_obs_size": sum(each),
            "each_subtask_obs_size": each,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": False,
            "each_subtask_name": names,
            "major_task_name": "carry",
            "has_extra": True,
        }

    def _steer_obs(self, rows):
        n = len(rows)
        if self.steer_zero:
            return torch.zeros(n, self.steer_dim(), device=self.device)
        h = self.humanoid_rows(self._humanoid_root_states)[rows]
        points = torch.cat([
            sp.window(self._gt_path[rows], self._arc_root[rows], self.steer_k, 0.4),
            sp.window(self._gt_path[rows], self._arc_box[rows], self.steer_k, 0.4),
        ], dim=1)
        heading = torch.atan2(
            2.0 * (h[:, 6] * h[:, 5] + h[:, 3] * h[:, 4]),
            1.0 - 2.0 * (h[:, 4] ** 2 + h[:, 5] ** 2))
        return sp.to_local(points, h[:, 0:2], heading).reshape(n, -1)

    def _compute_task_obs(self, env_ids=None):
        base = HumanoidMACarry._compute_task_obs(self, env_ids)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        teammate_dim = TEAMMATE_DIM * (self.num_agents - 1)
        carry = base[:, teammate_dim:teammate_dim + (CARRY_HI - CARRY_LO)]
        return torch.cat([self._steer_obs(rows), carry], dim=-1)
