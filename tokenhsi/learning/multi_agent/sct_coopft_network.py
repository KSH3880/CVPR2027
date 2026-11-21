from learning.multi_agent.mean_cooperation import MeanCooperation
from learning.multi_agent.stage2_push_network import Stage2PushBuilder


class SCTCoopFTBuilder(Stage2PushBuilder):
    class Network(Stage2PushBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            settings = params['cooperative_mlp']
            width = params['transformer']['num_features']
            self.coordination = MeanCooperation(width=width, **settings)
            self.critic_coordination = MeanCooperation(width=width, **settings)
