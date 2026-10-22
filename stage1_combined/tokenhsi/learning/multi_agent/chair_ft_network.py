from copy import deepcopy
import torch
from torch import nn
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder

class ChairTokenizer(nn.Module):
    """35D neural input; the separate 1D flag only routes chair vs old objects."""
    def __init__(self,base):
        super().__init__();self.base=base;self.chair=deepcopy(base)
        old=self.chair[0]
        self.chair[0]=nn.Linear(35,old.out_features,bias=old.bias is not None)
        with torch.no_grad():
            self.chair[0].weight.zero_();self.chair[0].weight[:,:30].copy_(old.weight)
            if old.bias is not None:self.chair[0].bias.copy_(old.bias)
    def forward(self,obs):
        original=obs[...,:30]
        chair_input=torch.cat([original,obs[...,31:36]],-1)
        return torch.where(obs[...,30:31]>.5,self.chair(chair_input),self.base(original))

class ChairFTBuilder(AMPMultiAgentBuilder):
    def build(self,name,**kwargs):return self.Network(self.params,**kwargs)
    class Network(AMPMultiAgentBuilder.Network):
        def __init__(self,params,**kwargs):
            actual=list(kwargs['scene_entity_sizes']);kwargs['scene_entity_sizes']=[actual[0],30,actual[2]]
            super().__init__(params,**kwargs);self.entity_sizes=actual
            for enc in (self.actor_encoder,self.critic_encoder):
                enc.entity_sizes=actual;enc.tokenizers[1]=ChairTokenizer(enc.tokenizers[1])

def transfer(model,checkpoint):
    state=model.state_dict();loaded=[]
    for name,target in state.items():
        source_name=name.replace('.tokenizers.1.base.','.tokenizers.1.').replace('.tokenizers.1.chair.','.tokenizers.1.')
        source=checkpoint['model'][source_name]
        if name.endswith('.tokenizers.1.chair.0.weight'):
            target.zero_();target[:,:30].copy_(source)
        elif target.shape==source.shape:target.copy_(source)
        else:raise ValueError('Unexpected transfer shape '+name)
        loaded.append(name)
    model.load_state_dict(state,strict=True);return loaded

def freeze_except_chair(model):
    model.requires_grad_(False)
    for enc in (model.a2c_network.actor_encoder,model.a2c_network.critic_encoder):enc.tokenizers[1].chair.requires_grad_(True)
    return [name for name,p in model.named_parameters() if p.requires_grad]
