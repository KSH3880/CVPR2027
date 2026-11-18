import torch

from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.before_object_alias import encoder_packet, goal_aligned_scene
from qualitative.warehouse_waypoint import ontop_waypoint_scene, role_waypoint_inputs, waypoint_scene


class WaypointBuilder(Stage2PushBuilder):
    class Network(Stage2PushBuilder.Network):
        waypoint = None

        def _humans(self, encoder, scene):
            if self.waypoint is None:
                return super()._humans(encoder, scene)
            leader, virtual_local, target = self.waypoint
            scene = waypoint_scene(scene, leader, virtual_local, self.num_agents, self.num_objects, target)
            width = 5*self.num_agents
            packet = encoder_packet(scene[:, -width:], self.num_agents, self.num_objects)
            return encoder(torch.cat((scene[:, :-width], packet), -1))

        def actor_with_follower_waypoint(self, obs, leader, virtual_local, active, target,
                                         follower, follower_active, follower_target):
            m, o = self.num_agents, self.num_objects
            scene, groups = obs[:, :-m], obs[:, -m:]
            views = [goal_aligned_scene(scene, m, o)]
            group_views = [groups]
            selections = []
            leader_rows = active.nonzero().flatten()
            if len(leader_rows):
                selected_target = None if target is None else target[leader_rows]
                views.append(waypoint_scene(scene[leader_rows], leader[leader_rows],
                                            virtual_local[leader_rows], m, o, selected_target))
                group_views.append(groups[leader_rows])
                selections.append((leader_rows, leader[leader_rows]))
            follower_rows = follower_active.nonzero().flatten()
            if len(follower_rows):
                selected_target = None if follower_target is None else follower_target[follower_rows]
                views.append(ontop_waypoint_scene(scene[follower_rows], follower[follower_rows],
                                                  virtual_local[follower_rows], m, o, selected_target))
                group_views.append(groups[follower_rows])
                selections.append((follower_rows, follower[follower_rows]))

            views = torch.cat(views)
            packet = views[:, -5*m:]
            redirected = encoder_packet(packet, m, o)
            encoded = self.actor_encoder(torch.cat((views[:, :-5*m], redirected), -1))
            contexts = self.coordination(encoded, None, packet, None, groups=torch.cat(group_views))
            humans = contexts[:len(obs)].clone()
            offset = len(obs)
            for rows, actors in selections:
                humans[rows, actors] = contexts[offset:offset+len(rows)][
                    torch.arange(len(rows), device=obs.device), actors]
                offset += len(rows)
            return self.action_head(humans).flatten(0, 1)

        def actor_with_box_only_waypoint(self, obs, metadata):
            m, o = self.num_agents, self.num_objects
            scene, groups = obs[:, :-m], obs[:, -m:]
            inputs, packet, rows, actors = role_waypoint_inputs(
                scene, metadata[:, [0, 5]].long(), metadata[:, [1, 6]], metadata[:, 2:5], m, o,
                box_only=True)
            encoded = self.actor_encoder(inputs)
            contexts = self.coordination(encoded, None, packet, None, groups=torch.cat((groups, groups[rows])))
            humans = contexts[:len(obs)].clone()
            humans[rows, actors] = contexts[len(obs)+torch.arange(len(rows), device=obs.device), actors]
            return self.action_head(humans).flatten(0, 1)


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
            relation_reward_mode=task._relation_cfg['mode'], relation_graph_spec=saved['env']['relationGraph'],
            device=task.device).to(task.device)
        model = torch.nn.Module()
        model.a2c_network = self.net
        model.load_state_dict(checkpoint['model'], strict=True)
        model.requires_grad_(False).eval()
        suffix = task.get_relation_suffix_size()-task._sensor_obs_width-getattr(task, '_waypoint_obs_width', 0)
        self.rms = SceneRunningMeanStd(task.get_scene_entity_sizes(), [4, 8, 4],
            normalized_sizes=task.get_scene_normalized_entity_sizes(), kinematic_size=7,
            extra_passthrough_size=suffix).to(task.device).eval()
        self.rms.load_state_dict(checkpoint['running_mean_std'], strict=True)

    @torch.no_grad()
    def actions(self, obs, leader, virtual_local, active, target=None, *,
                follower=None, follower_active=None, follower_target=None):
        obs = self.rms(obs[:, :self.width])
        if follower is not None:
            return self.net.actor_with_follower_waypoint(
                obs, leader, virtual_local, active, target, follower, follower_active,
                follower_target).clamp(-1, 1)
        self.net.waypoint = None
        original = self.net.eval_actor(obs)[0].reshape(len(obs), 4, -1)
        result = original.clone()
        if active.any():
            self.net.waypoint = (leader, virtual_local, target)
            guided = self.net.eval_actor(obs)[0].reshape_as(original)
            rows = active.nonzero().flatten()
            result[rows, leader[rows]] = guided[rows, leader[rows]]
        self.net.waypoint = None
        return result.flatten(0, 1).clamp(-1, 1)

    @torch.no_grad()
    def box_only_actions(self, obs, metadata):
        obs = self.rms(obs[:, :self.width])
        return self.net.actor_with_box_only_waypoint(obs, metadata).clamp(-1, 1)
