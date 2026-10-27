"""Native Table_Circle adaptation with existing HOLDING+AT rewards."""
import torch
from isaacgym.torch_utils import quat_from_angle_axis
from env.tasks.multi_agent.humanoid_chair_hold_at_ft import HumanoidChairHoldAtFT


class HumanoidTableHoldAtFT(HumanoidChairHoldAtFT):
    """Share chair FT contracts; original Table reset uses uniform random yaw."""

    def _reset_boxes(self, env_ids):
        n = len(env_ids); device = self.device
        direction = torch.randn(n, 2, device=device)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        distance = 1 + 4*torch.rand(n, 1, device=device)
        pos = torch.zeros(n, 3, device=device)
        pos[:,:2] = self._humanoid_root_states[env_ids,0,:2] + direction*distance
        pos[:,2] = self._object_ground_z[env_ids] + .05
        yaw = torch.rand(n, device=device) * (2 * torch.pi)
        axis = torch.zeros(n, 3, device=device); axis[:,2] = 1
        self._box_states[env_ids,0,:3] = pos
        self._box_states[env_ids,0,3:7] = quat_from_angle_axis(yaw, axis)
        self._box_states[env_ids,0,7:13] = 0
        self._platform_pos[env_ids] = self._platform_default_pos[env_ids]

