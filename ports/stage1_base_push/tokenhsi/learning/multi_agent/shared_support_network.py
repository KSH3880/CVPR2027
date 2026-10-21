import torch
from learning.multi_agent.stage2_network_builder import Stage2Builder


class SharedSupportBuilder(Stage2Builder):
    class Network(Stage2Builder.Network):
        def _encode_shared(self, encoder, coordination, obs):
            m = self.num_agents
            scene = obs[:, :-m]
            physical = obs[:, -m:].long()
            humans = encoder(scene)
            packet = obs.new_zeros(len(obs), m, 5)
            physical = torch.where(physical >= 0, physical, -1-torch.arange(m, device=obs.device))
            packet[..., 0] = 1
            packet[..., 1] = 2
            packet[..., 2] = torch.arange(m, device=obs.device)
            packet[..., 3] = physical
            packet[..., 4] = physical
            return coordination(humans, None, packet.flatten(1), None)

        def eval_actor(self, obs):
            humans = self._encode_shared(self.actor_encoder, self.coordination, obs)
            mu = self.action_head(humans).reshape(len(obs)*self.num_agents, -1)
            return mu, mu*0+self.sigma_act(self.sigma)

        def eval_critic(self, obs):
            humans = self._encode_shared(self.critic_encoder, self.critic_coordination, obs)
            return self.value_head(humans).reshape(len(obs)*self.num_agents, -1)
