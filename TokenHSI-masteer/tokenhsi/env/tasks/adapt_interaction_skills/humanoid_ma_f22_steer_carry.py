"""Two-agent execution shell for the frozen f22 carry+steer actor.

This is deliberately a baseline task, not the cooperative policy:

* every row receives the original f22 observation contract
  ``self(223) + dual-steer(24) + carry(42)``;
* there is no teammate token and no additional trainable module;
* the same frozen checkpoint is evaluated independently for both rows.

The A>=2 actor/object tensor plumbing comes from the already validated MA carry
backend.  Only its carry branch is enabled there; traj/sit/climb observations
and policies are not exposed by this class.
"""

import torch

from env.tasks.adapt_interaction_skills.humanoid_ma_carry import (
    HumanoidMACarry, CARRY_LO, CARRY_HI, TEAMMATE_DIM,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)
from tokenhsi.utils import steer_path as sp


class HumanoidMAF22SteerCarry(HumanoidMASteerCarry):
    """A=2 physical scene with an unchanged single-agent f22 policy per row."""

    def steer_dim(self):
        return 4 * self.steer_k

    def get_task_obs_size(self):
        # Raw policy observation.  The frozen old-carry token consumes the same
        # 42-D block internally, just as in the original f22 adapt task.
        return self.steer_dim() + (CARRY_HI - CARRY_LO)

    def get_multi_task_info(self):
        # During the validated MA carry backend's own observation calculation,
        # it needs its native four-task mask.  Outside that narrow call the
        # network sees exactly the original f22 three-token contract.
        if getattr(self, "_use_base_info", False):
            return super().get_multi_task_info()

        each = [self.steer_dim(), CARRY_HI - CARRY_LO, CARRY_HI - CARRY_LO]
        names = ["new_extra", "new_carry", "old_carry"]
        index = torch.cumsum(torch.tensor([0] + each), dim=0).to(self.device)
        mask = torch.zeros(len(each), sum(each), dtype=torch.bool, device=self.device)
        for i in range(len(each)):
            mask[i, index[i]:index[i + 1]] = True
        return {
            "onehot_size": len(each),
            "tota_subtask_obs_size": sum(each),
            "each_subtask_obs_size": each,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": False,
            "each_subtask_name": names,
            "major_task_name": "carry",
            "has_extra": True,
        }

    def _dual_steer_obs(self, rows):
        """Original f22 STEER_DUAL ordering and root-yaw coordinate frame."""
        n = len(rows)
        if self.steer_zero:
            return torch.zeros(n, self.steer_dim(), device=self.device)

        h = self.humanoid_rows(self._humanoid_root_states)[rows]
        path = self._gt_path[rows]
        pts = torch.cat([
            sp.window(path, self._arc_root[rows], self.steer_k, 0.4),
            sp.window(path, self._arc_box[rows], self.steer_k, 0.4),
        ], dim=1)
        heading = torch.atan2(
            2.0 * (h[:, 6] * h[:, 5] + h[:, 3] * h[:, 4]),
            1.0 - 2.0 * (h[:, 4] ** 2 + h[:, 5] ** 2))
        return sp.to_local(pts, h[:, 0:2], heading).reshape(n, -1)

    def _compute_task_obs(self, env_ids=None):
        # Call the A=2 carry backend directly, bypassing HumanoidMASteerCarry's
        # teammate and 12-D steering concatenation.
        base = HumanoidMACarry._compute_task_obs(self, env_ids)
        rows = self.all_rows() if env_ids is None else self.agent_rows(env_ids)
        teammate_dim = TEAMMATE_DIM * (self.num_agents - 1)
        carry = base[:, teammate_dim:teammate_dim + (CARRY_HI - CARRY_LO)]
        out = torch.cat([self._dual_steer_obs(rows), carry], dim=-1)
        assert out.shape[-1] == 66, out.shape
        return out