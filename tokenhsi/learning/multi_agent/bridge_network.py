from torch import nn

from learning.multi_agent.before_network import BeforeBuilder
from utils.bridge_spec import predecessor_relations


class BridgeBuilder(BeforeBuilder):
    class Network(BeforeBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            # PUSH-capable Stage1 checkpoints carry a fifth task embedding.
            for encoder in (self.actor_encoder, self.critic_encoder):
                encoder.edge_encoder.category_embed = nn.Embedding(5, 64)
                nn.init.trunc_normal_(encoder.edge_encoder.category_embed.weight, std=.02)

        def _coordinate(self, encoder, coordination, obs):
            # The last M values are each human's predecessor index (-1: none) for the BEFORE relation.
            m = self.num_agents
            humans = encoder(obs[:, :-m])
            relations = predecessor_relations(obs[:, -m:], m)
            return coordination(humans, None, obs[:, -6*m:-m], None, relations=relations)

        def eval_actor(self, obs):
            humans = self._coordinate(self.actor_encoder, self.coordination, obs)
            mu = self.action_head(humans).reshape(len(obs)*self.num_agents, -1)
            return mu, mu*0 + self.sigma_act(self.sigma)

        def eval_critic(self, obs):
            humans = self._coordinate(self.critic_encoder, self.critic_coordination, obs)
            return self.value_head(humans).reshape(len(obs)*self.num_agents, -1)
