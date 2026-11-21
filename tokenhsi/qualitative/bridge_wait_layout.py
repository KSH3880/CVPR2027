import torch

from env.tasks.multi_agent.humanoid_ma_bridge_follow_cap import HumanoidMABridgeFollowCap
from env.tasks.multi_agent.sensor_stage2_placement import SensorStage2Placement
from env.tasks.multi_agent import square_sensor_scene as scene


class BridgeWaitTask(HumanoidMABridgeFollowCap):
    def __init__(self, *args, positions, **kwargs):
        self.positions = positions
        super().__init__(*args, **kwargs)

    def _set_before_surfaces(self, ids, same):
        self._start_surface[ids, self._roles[ids, 1]] = 1
        super()._set_before_surfaces(ids, torch.zeros_like(same))

    def _place_bridge(self, ids, rows, occupied):
        env = ids[rows]
        opt = self.positions
        self._bridge_side[env] = opt.bridge_entry_side == 'left'
        SensorStage2Placement._place_bridge(self, ids, rows, occupied, place_crossers=False)
        yaw = self._structure_yaw[env, 1]
        shift, origin = self._structure_shift[env, 1], self._env_origins[env]
        world = lambda xy: scene.to_world(xy, yaw, shift, origin)
        if opt.bridge_distance is not None:
            start = self._bridge_landing[env].clone()
            start[:, 1] += opt.bridge_distance * (1 if opt.bridge_entry_side == 'left' else -1)
            desired = world(start)
            delta = desired - self._box_states[env, self._bridge_slot[env], :2]
            self._box_states[env, self._bridge_slot[env], :2] = desired
            for person in range(self.num_agents):
                selected = self._bridge_pushers[env, person]
                es = env[selected]
                if len(es):
                    roots = self._humanoid_root_states[es, person]
                    from utils.torch_utils import calc_heading
                    self._place_person(es, person, roots[:, :2] + delta[selected],
                        calc_heading(roots[:, 3:7]), torch.zeros(len(es), device=self.device))
        leader, follower = self._roles[env, 1], self._roles[env, 3]
        payload = self._agent_box_assignment[env, leader]
        sizes = self._box_size[env, payload]
        radius = sizes[:, :2].norm(dim=-1)/2
        centers, halves = self._surface_center[env, 1], self._surface_half[env, 1]
        margin = torch.maximum(radius+self._scene_cfg['edge_margin'],
            torch.full_like(radius, self._scene_cfg['human_margin']))
        human = centers[:, 0].clone()
        human[:, 0] += halves[:, 0, 0]-opt.leader_depth
        human[:, 1] -= opt.leader_lateral*(halves[:, 0, 1]-margin)
        box = human.clone()
        box[:, 0] += opt.pickup_distance
        goal = centers[:, 1].clone()
        goal[:, 0] += opt.goal_depth-halves[:, 1, 0]
        goal[:, 1] -= opt.goal_lateral*(halves[:, 1, 1]-radius-self._scene_cfg['edge_margin'])
        safe_human = ((human-centers[:, 0]).abs() <= halves[:, 0]-self._scene_cfg['human_margin']+1e-5).all(-1)
        safe_box = ((box-centers[:, 0]).abs() <= halves[:, 0]-(radius+self._scene_cfg['edge_margin'])[:, None]+1e-5).all(-1)
        safe_goal = ((goal-centers[:, 1]).abs() <= halves[:, 1]-(radius+self._scene_cfg['edge_margin'])[:, None]+1e-5).all(-1)
        separated = opt.pickup_distance >= radius+scene.HUMAN_RADIUS+scene.GROUP_PAD
        if not (safe_human & safe_box & safe_goal & separated).all():
            raise ValueError('Requested positions do not fit the platform or overlap: increase leader-depth/pickup-distance or adjust goal-depth')
        human_world, box_world, goal_world = map(world, (human, box, goal))
        height = self._surface_height[env, 1]
        self._place_person(env, leader, human_world, yaw, height)
        self._box_states[env, payload, :2] = box_world
        self._box_states[env, payload, 2] = origin[:, 2]+height+sizes[:, 2]/2
        self._tar_pos[env, leader, :2] = goal_world
        self._tar_pos[env, leader, 2] = self._box_states[env, payload, 2]
        radii = torch.stack((torch.full_like(radius, scene.HUMAN_RADIUS), radius), -1)
        occupied.add(rows, torch.stack((human, box), 1), radii, torch.ones_like(radii, dtype=torch.bool),
            torch.ones_like(leader))
        occupied.add(rows, goal[:, None], radius[:, None], torch.ones_like(radius[:, None], dtype=torch.bool),
            torch.full_like(leader, 2))
        entities = self._group_entities(ids)
        head = torch.arange(self.num_agents, device=self.device).expand(len(ids), -1)
        for person in range(self.num_agents):
            selected = rows[follower == person]
            if len(selected):
                self._place_group(ids, selected, person, head, entities, occupied)
