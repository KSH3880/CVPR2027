import os

import torch
from torch import nn
from learning.multi_agent.stage2_push_network import Stage2PushBuilder


class SensorStage2Builder(Stage2PushBuilder):
    class Network(Stage2PushBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.sensor_points = params['sensor_points']
            self.mask_sensor = os.environ.get('SENSOR_MASK') == '1'
            for name, head in (('actor', self.action_head), ('critic', self.value_head)):
                setattr(self, name+'_sensor', nn.Sequential(nn.Linear(self.sensor_points, 256), nn.ReLU(),
                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64)))
                projection = nn.Linear(64, head[0][0].out_features, bias=False)
                nn.init.zeros_(projection.weight)
                setattr(self, name+'_sensor_projection', projection)

        def _sensor_head(self, obs, name, encoder, coordination, head):
            width = self.num_agents*self.sensor_points
            humans = super()._encode(encoder, coordination, obs[:, :-width].contiguous())
            sensor = obs[:, -width:].reshape(-1, self.num_agents, self.sensor_points)
            addition = getattr(self, name+'_sensor_projection')(getattr(self, name+'_sensor')(sensor))
            if self.mask_sensor:
                addition = torch.zeros_like(addition)
            hidden = head[0][0](humans)+addition
            return head[1:](head[0][1:](hidden)).reshape(len(obs)*self.num_agents, -1)

        def eval_actor(self, obs):
            mu = self._sensor_head(obs, 'actor', self.actor_encoder, self.coordination, self.action_head)
            return mu, mu*0+self.sigma_act(self.sigma)

        def eval_critic(self, obs):
            return self._sensor_head(obs, 'critic', self.critic_encoder, self.critic_coordination, self.value_head)


def sensor_parameter(name):
    return name.startswith(('a2c_network.actor_sensor.', 'a2c_network.critic_sensor.',
        'a2c_network.actor_sensor_projection.', 'a2c_network.critic_sensor_projection.'))


def freeze_backbone(model):
    for name, parameter in model.named_parameters():
        frozen = name.startswith(('a2c_network.actor_encoder.', 'a2c_network.coordination.'))
        parameter.requires_grad_(not frozen and name != 'a2c_network.sigma')
    model.a2c_network.actor_encoder.eval()
    model.a2c_network.coordination.eval()


def transfer(model, checkpoint):
    graph = checkpoint['relation_experiment_config']['env']['relationGraph']
    if graph['human_relation_fusion'] != 'stage2_zerobg':
        raise ValueError('Sensor transfer requires integrated Stage2')
    target = model.state_dict()
    if set(checkpoint['model']) != {k for k in target if not sensor_parameter(k)}:
        raise ValueError('Sensor transfer base tensor keys differ')
    for key, value in checkpoint['model'].items():
        target[key].copy_(value)
    model.load_state_dict(target, strict=True)
    freeze_backbone(model)
