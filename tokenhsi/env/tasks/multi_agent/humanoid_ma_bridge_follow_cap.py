from isaacgym import gymapi
import torch

from env.tasks.multi_agent.bridge_follow_cap import CAP_FILTER, FollowCapMixin, load_body
from env.tasks.multi_agent.humanoid_ma_sensor_bridge_layout_cache import HumanoidMASensorBridgeLayoutCache
from env.tasks.multi_agent.edge_context_reward import owner_sum
from utils.bridge_waypoint import bridge_goal, update_switch, staged_progress
from utils.edge_context_spec import AT


class HumanoidMABridgeFollowCap(FollowCapMixin, HumanoidMASensorBridgeLayoutCache):
    def __init__(self, cfg, *args, **kwargs):
        bridge = cfg['env']['sensorScene']['bridge']
        self.bridge_physics = 'follow_cap'
        self.cap_lift = bridge['follow_cap']['lift']
        self._waypoint_spec = bridge['leader_waypoint']
        self._pusher_sampling = bridge.get('pusher_sampling')
        self._pusher_gaps = getattr(cfg['args'], 'pusher_gaps', None)
        assignment = getattr(cfg['args'], 'pusher_assignment', None)
        if assignment is not None:
            self._pusher_sampling = dict(self._pusher_sampling, assignment=assignment)
        super().__init__(cfg, *args, **kwargs)

    def _place_scene(self, ids):
        super()._place_scene(ids)
        if self._pusher_sampling is not None:
            from utils.bridge_pusher_sampling import sample_pushers
            sample_pushers(self, ids, self._pusher_sampling)

    def _reset_env_tensors(self, ids):
        if self._pusher_gaps is not None:
            from utils.bridge_pusher_sampling import set_pusher_gaps
            set_pusher_gaps(self, ids, self._pusher_gaps)
        super()._reset_env_tensors(ids)

    def _load_box_asset(self):
        super()._load_box_asset()
        cache = {}
        for env in range(self.num_envs):
            for slot in range(self._bridge_bank, self._bridge_bank+3):
                size = self._box_size[env, slot].tolist()
                index = env*self.num_physical_objects+slot
                density = float(self._box_density[index])
                key = (*size, density)
                if key not in cache:
                    options = gymapi.AssetOptions()
                    options.density = density
                    options.angular_damping = options.linear_damping = .01
                    options.max_angular_velocity = 100.
                    cache[key] = load_body(self.gym, self.sim,
                        self.cfg['args'].output_path+'/bridge_surface_assets', size, options,
                        self._scene_cfg['bridge']['top_surface']['thickness'])
                self._box_assets[index] = cache[key]

    def _build_box(self, env_id, object_id, env_ptr):
        if not self._bridge_bank <= object_id < self._bridge_bank+3:
            return super()._build_box(env_id, object_id, env_ptr)
        index = env_id*self.num_physical_objects+object_id
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(1000.+12*object_id, 1000., float(self._box_size[env_id, object_id, 2])/2)
        handle = self.gym.create_actor(env_ptr, self._box_assets[index], pose,
            f'box_{object_id}', env_id, 0, 0)
        self._box_handles.append(handle)
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        props[0].friction = self._joint['box']['friction']
        props[0].restitution = 0.
        props[0].filter |= CAP_FILTER
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

    def _init_bridge_curriculum(self):
        super()._init_bridge_curriculum()
        self._waypoint_switched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._waypoint_reference = torch.zeros(self.num_envs, device=self.device)
        self._waypoint_approach = torch.zeros(self.num_envs, device=self.device)

    def get_relation_suffix_size(self):
        return super().get_relation_suffix_size()+5

    def _waypoint_geometry(self, ids):
        leader = self._roles[ids, 1]
        payload = self._agent_box_assignment[ids, leader]
        sizes = self._box_size[ids, payload]
        bridge = self._box_states[ids, self._bridge_slot[ids]]
        goal = bridge_goal(bridge, self._box_size[ids, self._bridge_slot[ids]], sizes)
        switched, distance, radius = update_switch(self._box_states[ids, payload, :2],
            goal[:, :2], sizes, self._waypoint_spec['switch_margin'], self._waypoint_switched[ids])
        return goal, switched & ~self._before_same_platform[ids], distance, radius

    def _leader_at_edges(self, ids):
        graph = self.relation_runtime.graph
        return ((graph.edge_relation[ids] == AT) & graph.edge_valid[ids]
                & (graph.edge_owner[ids] == self._roles[ids, 1, None])).long().argmax(-1)

    def _reset_relation_history(self, ids):
        if self._pusher_gaps is not None and (ids == 0).any():
            people = self._bridge_pushers[0].nonzero().flatten().tolist()
            print('[viewer] env0 pusher rear-face gaps (agent ID, meters):',
                  list(zip(people, self._pusher_gaps)), flush=True)
        if self._pusher_sampling is not None:
            from utils.bridge_eval_assignment import randomize_pusher_assignment, restore_ordered_assignment
            if self._pusher_sampling['assignment'] == 'random':
                randomize_pusher_assignment(self, ids)
            else:
                restore_ordered_assignment(self, ids)
        super()._reset_relation_history(ids)
        self._waypoint_switched[ids] = False
        self._waypoint_reference[ids] = 0.
        self._waypoint_approach[ids] = 0.
        if self._waypoint_spec['enabled']:
            _, switched, _, _ = self._waypoint_geometry(ids)
            self._waypoint_switched[ids] = switched
            _, diag = self._evaluate_relations(ids)
            self._waypoint_reference[ids] = diag['progress'][
                torch.arange(len(ids), device=self.device), self._leader_at_edges(ids)]

    def _compute_clean_scene_obs(self, env_ids=None):
        obs = super()._compute_clean_scene_obs(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        goal, _, _, _ = self._waypoint_geometry(ids)
        active = self._waypoint_spec['enabled'] & ~self._waypoint_switched[ids] & ~self._before_same_platform[ids]
        metadata = torch.cat((self._roles[ids, 1, None].float(), active[:, None].float(),
                              goal-self._env_origins[ids]), -1)
        return torch.cat((obs, metadata), -1)

    def _adjust_relation_reward(self, result):
        if not self._waypoint_spec['enabled']:
            return
        ids = torch.arange(self.num_envs, device=self.device)
        _, switched, distance, radius = self._waypoint_geometry(ids)
        edges = self._leader_at_edges(ids)
        weight = self._relation_cfg['progress_reward_weight']
        original = result['progress_component'][ids, edges]
        final = original/weight
        fresh = switched & ~self._waypoint_switched
        self._waypoint_reference[fresh] = final[fresh]
        self._waypoint_switched.copy_(switched)
        if self._waypoint_spec['reward'] != 'staged':
            return
        sigma = self._scene_cfg['bridge']['before_training']['bridge_center_progress']['sigma']
        approach, final = staged_progress(distance, radius, switched, final, self._waypoint_reference, sigma)
        self._waypoint_approach.copy_(approach*(~self._before_same_platform))
        replacement = torch.where(self._before_same_platform, original, weight*final)
        result['progress_component'][ids, edges] = replacement
        result['total'] = result['state_component']+result['progress_component']+result['success_component']
        result['local_task_reward'] = owner_sum(result['total'], self.relation_runtime.graph)
        result['agent_task_reward'] = result['local_task_reward']

    def _compute_bridge_center_progress(self):
        if self._waypoint_spec['reward'] != 'staged':
            return super()._compute_bridge_center_progress()
        spec = self._scene_cfg['bridge']['before_training']['bridge_center_progress']
        reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        reward[torch.arange(self.num_envs, device=self.device), self._roles[:, 1]] = spec['weight']*self._waypoint_approach
        return reward
