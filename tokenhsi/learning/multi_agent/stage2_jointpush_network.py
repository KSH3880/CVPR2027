import torch

from learning.multi_agent.stage1_push_network import Stage1PushBuilder
from learning.multi_agent.human_coordination import HumanCoordination
from learning.multi_agent.stage2_network_builder import freeze_actor


class Stage2JointPushBuilder(Stage1PushBuilder):
    class Network(Stage1PushBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.coordination = HumanCoordination(shuffle_human_order=True)
            self.critic_coordination = HumanCoordination(shuffle_human_order=True)
            initialize = self.init_factory.create(name='default')
            for branch in (self.coordination, self.critic_coordination):
                for layer in branch.modules():
                    if isinstance(layer, torch.nn.Linear):
                        initialize(layer.weight)
                        torch.nn.init.zeros_(layer.bias)
                branch.init_identity()

        def _encode(self, encoder, coordination, obs):
            m = self.num_agents
            scene, groups = obs[:, :-m], obs[:, -m:]
            humans = encoder(scene)
            packet = obs.new_zeros(len(obs), m, 5)
            packet[..., 0] = 1.
            packet[..., 1] = 4.
            packet[..., 2] = torch.arange(m, device=obs.device)
            packet[..., 3] = groups
            packet[..., 4] = groups
            return coordination(humans, None, packet.flatten(1), None)

        def eval_actor(self, obs):
            humans = self._encode(self.actor_encoder, self.coordination, obs)
            mu = self.action_head(humans).reshape(len(obs)*self.num_agents, -1)
            return mu, mu*0+self.sigma_act(self.sigma)

        def eval_critic(self, obs):
            humans = self._encode(self.critic_encoder, self.critic_coordination, obs)
            return self.value_head(humans).reshape(len(obs)*self.num_agents, -1)


def transfer_stage1(model, checkpoint):
    source, target = checkpoint['model'], model.state_dict()
    copied, added = [], []
    for name, value in target.items():
        if name.startswith(('a2c_network.coordination.', 'a2c_network.critic_coordination.')):
            added.append(name)
            continue
        if name not in source or value.shape != source[name].shape:
            raise ValueError('Incompatible Stage1 PUSH parameter: '+name)
        value.copy_(source[name])
        copied.append(name)
    if set(source)-set(target):
        raise ValueError('Stage1 parameters would be discarded')
    model.load_state_dict(target, strict=True)
    freeze_actor(model)
    return dict(source_epoch=checkpoint.get('epoch'), copied=copied, added=added)
