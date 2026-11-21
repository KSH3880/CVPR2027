import torch

from learning.multi_agent.sensor_stage2_network import SensorStage2Builder
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from utils.before_object_alias import encoder_packet
from utils.bridge_waypoint import waypoint_scene


class BridgeWaypointBuilder(SensorStage2Builder):
    class Network(SensorStage2Builder.Network):
        def _sensor_head(self, obs, name, encoder, coordination, head):
            metadata, obs = obs[:, -5:], obs[:, :-5]
            width = self.num_agents*self.sensor_points
            scene_obs = obs[:, :-width].contiguous()
            humans = Stage2PushBuilder.Network._encode(self, encoder, coordination, scene_obs)
            rows = (metadata[:, 1] > 0).nonzero().flatten()
            if len(rows):
                leaders = metadata[rows, 0].long()
                scene, groups = scene_obs[rows, :-self.num_agents], scene_obs[rows, -self.num_agents:]
                scene = waypoint_scene(scene, leaders, metadata[rows, 2:], self.num_agents, self.num_objects)
                packet = scene[:, -5*self.num_agents:]
                encoded = encoder(torch.cat((scene[:, :-5*self.num_agents],
                    encoder_packet(packet, self.num_agents, self.num_objects)), -1))
                guided = coordination(encoded, None, packet, None, groups=groups)
                humans = humans.clone()
                humans[rows, leaders] = guided[torch.arange(len(rows), device=obs.device), leaders]
            sensor = obs[:, -width:].reshape(-1, self.num_agents, self.sensor_points)
            addition = getattr(self, name+'_sensor_projection')(getattr(self, name+'_sensor')(sensor))
            if self.mask_sensor:
                addition = torch.zeros_like(addition)
            hidden = head[0][0](humans)+addition
            return head[1:](head[0][1:](hidden)).reshape(len(obs)*self.num_agents, -1)
