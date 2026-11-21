import torch

from learning.multi_agent.sensor_stage2_network import SensorStage2Builder
from utils.before_object_alias import encoder_packet
from utils.bridge_waypoint import waypoint_scene


class BridgeWaypointSinglePassBuilder(SensorStage2Builder):
    class Network(SensorStage2Builder.Network):
        def _sensor_head(self, obs, name, encoder, coordination, head):
            metadata, obs = obs[:, -5:], obs[:, :-5]
            width = self.num_agents*self.sensor_points
            scene_obs = obs[:, :-width]
            scene, groups = scene_obs[:, :-self.num_agents], scene_obs[:, -self.num_agents:]
            scene = waypoint_scene(scene, metadata[:, 0].long(), metadata[:, 2:],
                self.num_agents, self.num_objects, active=metadata[:, 1] > 0)
            packet = scene[:, -5*self.num_agents:]
            humans = encoder(torch.cat((scene[:, :-5*self.num_agents],
                encoder_packet(packet, self.num_agents, self.num_objects)), -1),
                object_valid=getattr(self, 'object_valid', None))
            humans = coordination(humans, None, packet, None, groups=groups)
            sensor = obs[:, -width:].reshape(-1, self.num_agents, self.sensor_points)
            addition = getattr(self, name+'_sensor_projection')(getattr(self, name+'_sensor')(sensor))
            if self.mask_sensor:
                addition = torch.zeros_like(addition)
            hidden = head[0][0](humans)+addition
            return head[1:](head[0][1:](hidden)).reshape(len(obs)*self.num_agents, -1)
