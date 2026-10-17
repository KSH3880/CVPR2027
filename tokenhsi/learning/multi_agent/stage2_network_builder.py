from copy import deepcopy

import torch

from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.human_coordination import HumanCoordination
from learning.multi_agent.grasp_view import encode_grasp_views
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingFusion


class Stage2Builder(AMPMultiAgentBuilder):
    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)

    class Network(AMPMultiAgentBuilder.Network):
        def __init__(self, params, **kwargs):
            graph = kwargs['relation_graph_spec']
            base = deepcopy(graph)
            base['sampler'] = 'two_agent_four_object_independent_unified'
            base['edge_capacity'] = 4
            kwargs['relation_graph_spec'] = base
            super().__init__(params, **kwargs)
            self.relation_graph_spec = graph
            for encoder in (self.actor_encoder, self.critic_encoder):
                encoder.relation_graph_spec = graph
                encoder.context_fusion = TaskTypeEmbeddingFusion(merge_shared=True)
            self.coordination = HumanCoordination(shuffle_human_order=True)
            self.critic_coordination = HumanCoordination(shuffle_human_order=True)
            initialize = self.init_factory.create(name="default")
            for branch in (self.coordination, self.critic_coordination):
                for layer in branch.modules():
                    if isinstance(layer, torch.nn.Linear):
                        initialize(layer.weight)
                        torch.nn.init.zeros_(layer.bias)
            self.coordination.init_identity()
            self.critic_coordination.init_identity()

        def _humans(self, encoder, obs):
            return encode_grasp_views(encoder, obs)

        def eval_actor(self, obs):
            humans = self._humans(self.actor_encoder, obs)
            humans = self.coordination(humans, None, obs[:, -5*self.num_agents:], None)
            mu = self.action_head(humans).reshape(len(obs)*self.num_agents, -1)
            return mu, mu*0 + self.sigma_act(self.sigma)

        def eval_critic(self, obs):
            humans = self._humans(self.critic_encoder, obs)
            humans = self.critic_coordination(humans, None, obs[:, -5*self.num_agents:], None)
            return self.value_head(humans).reshape(len(obs)*self.num_agents, -1)


def transfer_stage1(model, checkpoint):
    from utils.relation_task_spec import check_checkpoint_metadata, checkpoint_metadata
    from pathlib import Path
    import yaml
    env = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    check_checkpoint_metadata(checkpoint, checkpoint_metadata(env['relationReward']))
    saved_env = checkpoint.get('relation_experiment_config', {}).get('env', {})
    if saved_env.get('box', {}).get('reset', {}).get('platformPhysics') != 'fixed_warmup_v1':
        raise ValueError('Stage2 transfer requires a fixed-platform Stage1 checkpoint')
    source = checkpoint['model']
    target = model.state_dict()
    copied, added = [], []
    for key in target:
        if key.startswith(('a2c_network.coordination.', 'a2c_network.critic_coordination.')):
            added.append(key)
            continue
        if key not in source or source[key].shape != target[key].shape:
            raise ValueError('Incompatible Stage1 parameter: ' + key)
        target[key].copy_(source[key])
        copied.append(key)
    if set(source) - set(target):
        raise ValueError('Stage1 tensors would be discarded')
    model.load_state_dict(target, strict=True)
    freeze_actor(model)
    return dict(source_epoch=checkpoint.get('epoch'), copied=copied, added=added)


def freeze_actor(model):
    for name, parameter in model.named_parameters():
        if name.startswith('a2c_network.actor_encoder.'):
            parameter.requires_grad_(False)
    model.a2c_network.actor_encoder.eval()
