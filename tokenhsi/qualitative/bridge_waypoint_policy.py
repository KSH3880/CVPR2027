import torch

from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.before_object_alias import encoder_packet
from qualitative.bridge_waypoint import waypoint_scene


class WaypointBuilder(Stage2PushBuilder):
    class Network(Stage2PushBuilder.Network):
        waypoint = None

        def _humans(self, encoder, scene):
            if self.waypoint is None:
                return super()._humans(encoder, scene)
            leader, local_goal = self.waypoint
            scene = waypoint_scene(scene, leader, local_goal, self.num_agents, self.num_objects)
            width = 5*self.num_agents
            packet = encoder_packet(scene[:, -width:], self.num_agents, self.num_objects)
            return encoder(torch.cat((scene[:, :-width], packet), -1))


class WaypointPolicy:
    def __init__(self, task, checkpoint):
        saved = checkpoint['relation_experiment_config']
        if saved['experiment']['name'] != 'stage2_distill_900_v4':
            raise ValueError('This toy viewer requires the original Distill900V4 checkpoint')
        builder = WaypointBuilder()
        builder.load(saved['train']['params']['network'])
        self.width = saved['observation_size']
        self.net = builder.build('amp', actions_num=task.num_dof, input_shape=(self.width,),
            amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=4, num_objects=8,
            observation_mode='clean_scene', scene_entity_sizes=task.get_scene_entity_sizes(),
            scene_kinematic_size=7, scene_arena_scale=task.get_scene_arena_scale(),
            relation_reward_mode=task._relation_cfg['mode'], relation_graph_spec=task._relation_graph_spec,
            device=task.device).to(task.device)
        model = torch.nn.Module()
        model.a2c_network = self.net
        model.load_state_dict(checkpoint['model'], strict=True)
        model.requires_grad_(False).eval()
        self.rms = SceneRunningMeanStd(task.get_scene_entity_sizes(), [4, 8, 4],
            normalized_sizes=task.get_scene_normalized_entity_sizes(), kinematic_size=7,
            extra_passthrough_size=self.width-16*7-sum(size*count for size, count in
                zip(task.get_scene_entity_sizes(), (4, 8, 4)))).to(task.device).eval()
        self.rms.load_state_dict(checkpoint['running_mean_std'], strict=True)

    @torch.no_grad()
    def actions(self, obs, leader, goal_local, active):
        obs = self.rms(obs[:, :self.width])
        self.net.waypoint = None
        original = self.net.eval_actor(obs)[0].reshape(len(obs), 4, -1)
        result = original.clone()
        if active.any():
            self.net.waypoint = (leader, goal_local)
            guided = self.net.eval_actor(obs)[0].reshape_as(original)
            rows = active.nonzero().flatten()
            result[rows, leader[rows]] = guided[rows, leader[rows]]
        self.net.waypoint = None
        return result.flatten(0, 1).clamp(-1, 1)
