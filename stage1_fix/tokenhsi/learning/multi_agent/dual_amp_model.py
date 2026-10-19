"""Independent full/masked discriminators; TeamHOI adaptation, see third_party/teamhoi."""
import torch
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from learning.amp_models import ModelAMPContinuous
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder


class DualAMPBuilder(AMPMultiAgentBuilder):
    def build(self,name,**kwargs):return self.Network(self.params,**kwargs)

    class Network(AMPMultiAgentBuilder.Network):
        def _build_disc(self,input_shape):
            super()._build_disc(input_shape)
            full_mlp,full_logits=self._disc_mlp,self._disc_logits
            super()._build_disc(input_shape)  # Independent initialization, not shared/copy weights.
            self._disc_mlp_masked,self._disc_logits_masked=self._disc_mlp,self._disc_logits
            self._disc_mlp,self._disc_logits=full_mlp,full_logits

        def eval_disc_masked(self,obs):return self._disc_logits_masked(self._disc_mlp_masked(obs))

        def get_disc_logit_weights_masked(self):return self._disc_logits_masked.weight.flatten()

        def get_disc_weights_masked(self):
            return [m.weight.flatten() for m in self._disc_mlp_masked.modules() if isinstance(m,torch.nn.Linear)]+[self._disc_logits_masked.weight.flatten()]


class DualAMPModel(ModelAMPContinuous):
    def build(self,config):return self.Network(self.network_builder.build('amp',**config))

    class Network(ModelAMPContinuous.Network):
        def forward(self,inputs):
            result=ModelA2CContinuousLogStd.Network.forward(self,inputs)
            if inputs.get('is_train',True):
                for suffix,method in [('',self.a2c_network.eval_disc),('_masked',self.a2c_network.eval_disc_masked)]:
                    for key,label in [('amp_obs','disc_agent_logit'),('amp_obs_replay','disc_agent_replay_logit'),('amp_obs_demo','disc_demo_logit')]:
                        value=inputs[key+suffix]
                        result[label+suffix]=None if value is None else method(value)
            return result
