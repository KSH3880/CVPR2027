"""Native Chair carry reset and reward with the frozen 78k scene policy."""
import torch
from env.tasks.multi_agent.humanoid_chair_hold_at_ft import HumanoidChairHoldAtFT
from env.tasks.multi_agent.humanoid_table_tokenhsi_ft import HumanoidTableTokenHSIFT


class HumanoidChairTokenHSIFT(HumanoidTableTokenHSIFT):
    def _reset_boxes(self, env_ids):
        # Preserve the native Chair front-facing reset; reward/termination kernels
        # are shared by Chair and Table in the original TokenHSI implementation.
        HumanoidChairHoldAtFT._reset_boxes(self, env_ids)
        if not hasattr(self, '_native_source'):
            self._native_source = torch.zeros(self.num_envs, 3, device=self.device)
        self._native_source[env_ids] = self._box_states[env_ids, 0, :3]
