import torch

from env.tasks.multi_agent.humanoid_ma_sensor_bridge_before import HumanoidMASensorBridgeBefore
from env.tasks.multi_agent.sensor_stage2_placement import SensorStage2Placement
from env.tasks.multi_agent import square_sensor_scene as scene


class BridgeWaypointTask(HumanoidMASensorBridgeBefore):
    def _place_bridge(self, ids, rows, occupied):
        SensorStage2Placement._place_bridge(self, ids, rows, occupied, place_crossers=False)
        env = ids[rows]
        index = torch.arange(len(env), device=self.device)
        leader, follower = self._roles[env, 1], self._roles[env, 3]
        payload = self._agent_box_assignment[env, leader]
        size = self._box_size[env, payload]
        radius = size[:, :2].norm(dim=-1)/2
        start = self._start_surface[env, leader]-1
        centers = self._surface_center[env, 1]
        halves = self._surface_half[env, 1]
        landing = self._bridge_landing[env]
        direction = torch.sign(centers[index, start, 0]-landing[:, 0])
        source_width = 2*halves[index, start, 0]
        dest_width = 2*halves[index, 1-start, 0]
        low = radius+1.5
        high = torch.minimum(radius+3., source_width-1.8)
        payload_depth = low+(high-low)*torch.rand_like(low)
        human_depth = payload_depth+1.+(source_width-.8-payload_depth-1.)*torch.rand_like(low)
        goal_high = torch.minimum(radius+3., dest_width-radius-.5)
        goal_depth = low+(goal_high-low)*torch.rand_like(low)
        human_xy, payload_xy, goal_xy = [landing.clone() for _ in range(3)]
        half_gap = self._bridge_gap[env]/2
        human_xy[:, 0] += direction*(half_gap+human_depth)
        payload_xy[:, 0] += direction*(half_gap+payload_depth)
        goal_xy[:, 0] -= direction*(half_gap+goal_depth)
        yaw = self._structure_yaw[env, 1]
        shift, origin = self._structure_shift[env, 1], self._env_origins[env]
        world = lambda xy: scene.to_world(xy, yaw, shift, origin)
        human_world, payload_world, goal_world = map(world, (human_xy, payload_xy, goal_xy))
        heading = torch.atan2(payload_world[:, 1]-human_world[:, 1], payload_world[:, 0]-human_world[:, 0])
        height = self._surface_height[env, 1]
        self._place_person(env, leader, human_world, heading, height)
        self._box_states[env, payload, :2] = payload_world
        self._box_states[env, payload, 2] = origin[:, 2]+height+size[:, 2]/2
        self._tar_pos[env, leader, :2] = goal_world
        self._tar_pos[env, leader, 2] = self._box_states[env, payload, 2]
        self._place_failed[env] |= (high < low) | (goal_high < low)
        radii = torch.stack((torch.full_like(radius, scene.HUMAN_RADIUS), radius), -1)
        occupied.add(rows, torch.stack((human_xy, payload_xy), 1), radii,
                     torch.ones_like(radii, dtype=torch.bool), start+1)
        occupied.add(rows, goal_xy[:, None], radius[:, None],
                     torch.ones_like(radius[:, None], dtype=torch.bool), 2-start)
        entities = self._group_entities(ids)
        head = torch.arange(self.num_agents, device=self.device).expand(len(ids), -1)
        for person in range(self.num_agents):
            selected = rows[follower == person]
            if len(selected):
                self._place_group(ids, selected, person, head, entities, occupied)
