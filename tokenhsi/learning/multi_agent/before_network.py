import torch

from learning.multi_agent.stage2_network_builder import Stage2Builder


class BeforeBuilder(Stage2Builder):
    class Network(Stage2Builder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            mode = self.relation_graph_spec.get('actor_head_adapter')
            if mode is None:
                return
            if mode != 'tokenhsi_internal_v1':
                raise ValueError('Unknown BEFORE actor_head_adapter')
            hidden = self.action_head[0]
            if (len(hidden) != 4 or not all(isinstance(hidden[i], torch.nn.Linear) for i in (0, 2))
                    or not all(isinstance(hidden[i], torch.nn.ReLU) for i in (1, 3))):
                raise ValueError('Internal adapters require two Linear/ReLU hidden layers')
            # Separate modules preserve every Stage1 action_head checkpoint key.
            # These are parallel linear corrections, summed BEFORE the existing ReLU.
            self.action_adapters = torch.nn.ModuleList([
                torch.nn.Linear(hidden[i].in_features, hidden[i].out_features) for i in (0, 2)])
            for adapter in self.action_adapters:
                torch.nn.init.zeros_(adapter.weight)
                torch.nn.init.zeros_(adapter.bias)

        def _actor_actions(self, humans):
            if not hasattr(self, 'action_adapters'):
                return super()._actor_actions(humans)
            hidden = self.action_head[0]
            for i, adapter in enumerate(self.action_adapters):
                humans = hidden[2*i+1](hidden[2*i](humans) + adapter(humans))
            return self.action_head[1](humans)

        def _humans(self, encoder, obs):
            return encoder(obs)
