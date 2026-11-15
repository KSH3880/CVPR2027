"""One five-task policy for BEFORE and coupled PUSH Stage2 training."""
import torch
from torch import nn
from learning.multi_agent.before_network import BeforeBuilder,transfer_stage1 as transfer_before
from learning.multi_agent.human_coordination import HumanCoordination
from utils.before_object_alias import encoder_packet


class Stage2PushBuilder(BeforeBuilder):
    class Network(BeforeBuilder.Network):
        def __init__(self,params,**kwargs):
            super().__init__(params,**kwargs)
            for encoder in (self.actor_encoder,self.critic_encoder):
                encoder.edge_encoder.category_embed=nn.Embedding(5,64)
                nn.init.trunc_normal_(encoder.edge_encoder.category_embed.weight,std=.02)
            self.coordination=HumanCoordination(shuffle_human_order=True,relations=4)
            self.critic_coordination=HumanCoordination(shuffle_human_order=True,relations=4)
            for branch in (self.coordination,self.critic_coordination):
                # Match the reference Stage2 initialization, including FFN input bias.
                for layer in branch.modules():
                    if isinstance(layer,nn.Linear) and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                branch.init_identity()

        def _humans(self,encoder,scene):
            # The packet itself identifies BEFORE dependencies. Coupled and solo
            # packets pass through unchanged; config flags must not select a path.
            width=5*self.num_agents
            packet=encoder_packet(scene[:,-width:],self.num_agents,self.num_objects)
            return encoder(torch.cat((scene[:,:-width],packet),-1))

        def _encode(self,encoder,coordination,obs):
            m=self.num_agents
            scene,groups=obs[:,:-m],obs[:,-m:]
            packet=scene[:,-5*m:]
            humans=self._humans(encoder,scene)
            return coordination(humans,None,packet,None,groups=groups)

        def eval_actor(self,obs):
            humans=self._encode(self.actor_encoder,self.coordination,obs)
            mu=self.action_head(humans).reshape(len(obs)*self.num_agents,-1)
            return mu,mu*0+self.sigma_act(self.sigma)

        def eval_critic(self,obs):
            humans=self._encode(self.critic_encoder,self.critic_coordination,obs)
            return self.value_head(humans).reshape(len(obs)*self.num_agents,-1)


def transfer_stage1(model,checkpoint):
    saved=checkpoint.get('relation_experiment_config',{}).get('env',{})
    if (not checkpoint.get('stage1_jointpush_contract') or not saved.get('jointPush')
            or saved.get('before') or saved.get('stage2Policy')):
        raise ValueError('Stage2 PUSH transfer requires a Stage1 PUSH checkpoint')
    return transfer_before(model,checkpoint)
