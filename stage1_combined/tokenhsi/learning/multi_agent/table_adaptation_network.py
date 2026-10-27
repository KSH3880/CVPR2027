"""Frozen 78k policy with an isolated action residual OR copied object tokenizer."""
import copy
import torch
from torch import nn
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder


class TableAdaptationBuilder(AMPMultiAgentBuilder):
    def build(self,name,**kwargs):
        return self.Network(self.params,**kwargs)

    class Network(AMPMultiAgentBuilder.Network):
        def __init__(self,params,**kwargs):
            super().__init__(params,**kwargs)
            cfg=params['table_adaptation'];self.adaptation_mode=cfg['mode']
            if self.adaptation_mode=='residual':
                # Do not perturb the common RNG stream or initial environments.
                with torch.random.fork_rng(devices=[]):
                    self.action_residual=nn.Sequential(nn.Linear(params['transformer']['num_features'],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,kwargs['actions_num']))
                nn.init.zeros_(self.action_residual[-1].weight);nn.init.zeros_(self.action_residual[-1].bias)
            elif self.adaptation_mode=='tokenizer':
                self.adapt_object_tokenizer=copy.deepcopy(self.actor_encoder.tokenizers[1])
                self.actor_encoder.tokenizers[1].register_forward_hook(self._adapt_token)
            else:raise ValueError(self.adaptation_mode)

        def _adapt_token(self,module,inputs,output):
            return self.adapt_object_tokenizer(inputs[0])

        def initialize_adapter(self):
            if self.adaptation_mode=='tokenizer':
                self.adapt_object_tokenizer.load_state_dict(self.actor_encoder.tokenizers[1].state_dict(),strict=True)

        def eval_actor(self,obs):
            if self.adaptation_mode=='tokenizer':return super().eval_actor(obs)
            with torch.no_grad():
                encoded=self.actor_encoder(obs)
                base=self.action_head(encoded)
            residual=self.action_residual(encoded.detach())
            mu=(base+residual).reshape(obs.shape[0]*self.num_agents,-1)
            self.last_residual_rms=float(residual.detach().square().mean().sqrt())
            return mu,mu*0.+self.sigma_act(self.sigma)
