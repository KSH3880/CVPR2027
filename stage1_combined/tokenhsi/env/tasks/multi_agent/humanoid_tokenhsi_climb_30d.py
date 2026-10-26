"""Same CLIMB environment, original 30D scene inputs, no chair tokenizer."""
from env.tasks.multi_agent.humanoid_tokenhsi_climb_ft import HumanoidTokenHSIClimbFT
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
class HumanoidTokenHSIClimb30FT(HumanoidTokenHSIClimbFT):
    def get_obs_size(self):return HumanoidMACarry.get_obs_size(self)
    def get_scene_entity_sizes(self):return HumanoidMACarry.get_scene_entity_sizes(self)
    def _compute_clean_scene_obs(self,env_ids=None):return HumanoidMACarry._compute_clean_scene_obs(self,env_ids)
