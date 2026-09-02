"""Box stacking experiment using the existing non-F22 MA-steer policy contract.

The shared-world stacking state machine, object layout, phase commands and
rewards are reused from ``HumanoidMAF22StackCarry``.  Policy observations are
deliberately restored to ``HumanoidMASteerCarry``'s 340-D contract so an
existing MA-steer checkpoint can be evaluated without changing its network:

    self(223) + teammate(21) + steer(12) + carry(42) + carry(42)

No coordination token is exposed in this baseline.  Phase changes are visible
only through the existing steer/carry goals and teammate observation.
"""

from env.tasks.adapt_interaction_skills.humanoid_ma_f22_stack_carry import (
    HumanoidMAF22StackCarry,
)
from env.tasks.adapt_interaction_skills.humanoid_ma_steer_carry import (
    HumanoidMASteerCarry,
)


class HumanoidMAStackCarry(HumanoidMAF22StackCarry):
    """Stacking world with the unchanged non-F22 MA-steer observation ABI."""

    def steer_dim(self):
        # HumanoidMAF22SteerCarry changes this to a dual 24-D window.  The
        # existing MA-steer checkpoint was trained with one 6-point 2-D window.
        return 2 * self.steer_k

    def get_task_obs_size(self):
        return HumanoidMASteerCarry.get_task_obs_size(self)

    def get_multi_task_info(self):
        return HumanoidMASteerCarry.get_multi_task_info(self)

    def _compute_task_obs(self, env_ids=None):
        # Bypass both F22 observation overrides and preserve teammate input.
        return HumanoidMASteerCarry._compute_task_obs(self, env_ids)
