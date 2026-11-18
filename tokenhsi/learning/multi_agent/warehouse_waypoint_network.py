"""PPO-safe mixed real/waypoint views; all guidance comes from stored observations."""
import torch
from qualitative.warehouse_waypoint import role_waypoint_inputs
from learning.multi_agent.sensor_stage2_network import SensorStage2Builder
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from utils.before_object_alias import encoder_packet, goal_aligned_scene
from utils.warehouse_waypoint import OBS_WIDTH, ROLE_OBS_WIDTH, waypoint_scene


class WarehouseWaypointBuilder(SensorStage2Builder):
    class Network(SensorStage2Builder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.waypoint_obs_width = params.get('waypoint_obs_width', OBS_WIDTH)

        def _waypoint_humans(self, clean, metadata, encoder, coordination):
            """Encode isolated real and role-specific scenes in one batch."""
            m = self.num_agents
            scene, groups = clean[:, :-m], clean[:, -m:]
            if self.waypoint_obs_width == ROLE_OBS_WIDTH:
                inputs, packet, rows, actors = role_waypoint_inputs(
                    scene, metadata[:, [0, 5]].long(), metadata[:, [1, 6]], metadata[:, 2:5],
                    m, self.num_objects)
            else:
                rows = ((metadata[:, 1] == 0) | (metadata[:, 1] == 2)).nonzero().flatten()
                actors = metadata[rows, 0].long()
                views = goal_aligned_scene(scene, m, self.num_objects)
                if len(rows):
                    guided = waypoint_scene(scene[rows], actors, metadata[rows, 2:],
                                            m, self.num_objects, metadata[rows, 1] == 2)
                    views = torch.cat((views, guided))
                packet = views[:, -5*m:]
                inputs = torch.cat((views[:, :-5*m], encoder_packet(packet, m, self.num_objects)), -1)
            valid = getattr(self, 'object_valid', None)
            if len(rows):
                groups = torch.cat((groups, groups[rows]), 0)
                if valid is not None:
                    valid = torch.cat((valid, valid[rows]), 0)
            encoded = encoder(inputs, object_valid=valid)
            contexts = coordination(encoded, None, packet, None, groups=groups)
            humans = contexts[:len(clean)].clone()
            if len(rows):
                humans[rows, actors] = contexts[len(clean)+torch.arange(len(rows), device=clean.device), actors]
            return humans

        @torch.no_grad()
        def encode_actor_features(self, obs):
            if any(p.requires_grad for branch in (self.actor_encoder, self.coordination) for p in branch.parameters()):
                raise RuntimeError('Actor feature reuse requires a frozen encoder and coordination')
            width = self.num_agents*self.sensor_points
            suffix = self.waypoint_obs_width
            return self._waypoint_humans(obs[:, :-suffix-width].contiguous(), obs[:, -suffix:],
                                          self.actor_encoder, self.coordination)

        def _head_with_features(self, base, humans, name, head):
            width = self.num_agents*self.sensor_points
            sensor = base[:, -width:].reshape(-1, self.num_agents, self.sensor_points)
            addition = getattr(self, name+'_sensor_projection')(getattr(self, name+'_sensor')(sensor))
            if self.mask_sensor:
                addition = torch.zeros_like(addition)
            hidden = head[0][0](humans)+addition
            return head[1:](head[0][1:](hidden)).reshape(len(base)*self.num_agents, -1)

        def _sensor_head(self, obs, name, encoder, coordination, head):
            suffix = self.waypoint_obs_width
            base, metadata = obs[:, :-suffix], obs[:, -suffix:]
            width = self.num_agents*self.sensor_points
            humans = self._waypoint_humans(base[:, :-width].contiguous(), metadata, encoder, coordination)
            return self._head_with_features(base, humans, name, head)

        def actor_from_features(self, obs, features):
            mu = self._head_with_features(obs[:, :-self.waypoint_obs_width], features, 'actor', self.action_head)
            return mu, mu*0+self.sigma_act(self.sigma)

        def forward(self, obs_dict):
            features = obs_dict.get('frozen_actor_features')
            if features is None:
                return super().forward(obs_dict)
            obs = obs_dict['obs']
            return self.actor_from_features(obs, features) + (self.eval_critic(obs), obs_dict.get('rnn_states'))
