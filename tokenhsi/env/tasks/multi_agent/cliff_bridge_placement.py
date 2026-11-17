import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.sensor_stage2_placement import SensorStage2Placement
from env.tasks.multi_agent.stage2_layout import invalid_before_layout
from utils.cliff_bridge_layout import crossing_layout
from utils.cliff_bridge_spec import CONTROLLED, COOPERATIVE, FREE


class CliffBridgePlacement:
    def _reset_boxes(self, ids):
        super()._reset_boxes(ids)
        self._box_states[ids, self._controlled_bank:, :2] = self._env_origins[ids, None, :2]+1000.

    def _place_bridge(self, ids, rows, occupied):
        branch = self._bridge_branch[ids[rows]]
        cooperative = rows[branch == COOPERATIVE]
        if len(cooperative):
            SensorStage2Placement._place_bridge(self, ids, cooperative, occupied)
        selected = rows[branch != COOPERATIVE]
        if len(selected):
            self._place_crossers(ids[selected])

    def _place_crossers(self, env):
        cfg = self._scene_cfg['bridge']
        n, m = len(env), self.num_agents
        rows = torch.arange(n, device=self.device)
        people = torch.arange(m, device=self.device)[None].expand(n, -1)
        origin = self._env_origins[env]
        yaw, shift = self._structure_yaw[env, 1], self._structure_shift[env, 1]
        world = lambda xy: scene.to_world(xy, yaw, shift, origin)
        size = self._box_size[env, self._bridge_slot[env]]
        free = self._bridge_branch[env] == FREE
        layout = scene.bridge_layout(self._bridge_half_a[env], self._bridge_half_b[env],
            self._bridge_far[env], self._bridge_gap[env], size, cfg,
            torch.zeros(n, dtype=torch.bool, device=self.device),
            torch.ones(n, dtype=torch.bool, device=self.device))
        self._bridge_landing[env] = layout['landing']
        start = torch.cat((world(layout['start']), origin[:, 2:3]+size[:, 2:3]/2), -1)
        goal = torch.cat((world(layout['landing']), start[:, 2:3]), -1)
        bridge = torch.zeros(n, 13, device=self.device)
        bridge[:, :3] = goal
        bridge[:, 3:7] = scene.yaw_quat(yaw)
        self._box_states[env, self._bridge_slot[env]] = bridge
        self._bridge_landing_world[env] = goal
        controlled = ~free
        if controlled.any():
            self._reset_controller(env[controlled], start[controlled], goal[controlled])
        for person in range(m):
            selected = self._bridge_pushers[env, person]
            es = env[selected]
            if len(es):
                xy = world(layout['pusher'])[selected]
                self._place_person(es, person, xy, self._heading_to(xy, goal[selected, :2]),
                    torch.zeros(len(es), device=self.device))
                self._tar_pos[es, person] = goal[selected]
        source_surface = self._start_surface[env].gather(1, self._roles[env, 1:2]).flatten()-1
        dest = 1-source_surface
        centers, halves = self._surface_center[env, 1], self._surface_half[env, 1]
        source = self._agent_box_assignment[env]
        sizes = self._box_size[env]
        radii = sizes[rows[:, None], source, :2].norm(dim=-1)/2
        second = sizes[rows[:, None], 3*m+people, :2].norm(dim=-1)/2
        second = torch.where(self._templates[env] == 4, second, radii)
        payload, target, humans, failed = crossing_layout(centers[rows, source_surface], halves[rows, source_surface],
            centers[rows, dest], halves[rows, dest], layout['landing'], size, self._bridge_gap[env],
            self._templates[env], self._before_predecessor[env], torch.stack((radii, second), -1),
            self._cross_phase[env], self._cross_progress[env], self._before_near_pairs[env],
            self._bridge_curriculum, self.cfg['env']['before']['waiting_penalty']['radius_buffer'],
            distance_alpha=self._compact_distance[env] if hasattr(self, '_compact_distance') else None,
            boundary=self._compact_kind[env] if hasattr(self, '_compact_kind') else None)
        self._place_failed[env[failed]] = True
        height = self._surface_height[env, 1]
        for person in range(m):
            active = self._crosser_mask[env, person]
            es, primary = env[active], source[active, person]
            if not len(es):
                continue
            box_xy, target_xy, human_xy = world(payload[:, person])[active], world(target[:, person])[active], world(humans[:, person])[active]
            z = origin[active, 2]+height[active]
            self._box_states[es, primary, :2] = box_xy
            self._box_states[es, primary, 2] = z+sizes[active, primary, 2]/2
            at = self._templates[es, person] == 3
            self._tar_pos[es[at], person, :2] = target_xy[at]
            self._tar_pos[es[at], person, 2] = (z+sizes[active, primary, 2]/2)[at]
            top = (self._templates[es, person] == 4) & (self._before_predecessor[es, person] < 0)
            support = 3*m+person
            self._box_states[es[top], support, :2] = target_xy[top]
            self._box_states[es[top], support, 2] = z[top]+sizes[active, support, 2][top]/2
            lift = torch.where(self._cross_phase[es, person] > 0, size[active, 2], height[active])
            aim = torch.where((self._before_predecessor[es, person] >= 0)[:, None], target_xy, box_xy)
            moved = self._place_person(es, person, human_xy, self._heading_to(human_xy, aim), lift)
            self._move_neutral_goal(es, person, moved, lift)

    def _place_scene(self, ids):
        super()._place_scene(ids)
        self._place_failed[ids] |= invalid_before_layout(self, ids)
        cliff = ids[self._scene_ids[ids] == 0]
        if not len(cliff):
            return
        m = self.num_agents
        primary = self._agent_box_assignment[cliff]
        supports = 3*m+torch.arange(m, device=self.device)[None].expand(len(cliff), -1)
        slots = torch.cat((primary, supports), -1)
        rows = torch.arange(len(cliff), device=self.device)[:, None]
        boxes = self._box_states[cliff]
        xy = boxes[rows, slots, :2]
        radii = self._box_size[cliff][rows, slots, :2].norm(dim=-1)/2
        valid = torch.cat((torch.ones_like(primary, dtype=torch.bool), self._templates[cliff] == 4), -1)
        pairs = torch.triu(torch.ones(2*m, 2*m, dtype=torch.bool, device=self.device), diagonal=1)
        clash = (xy[:, :, None]-xy[:, None]).norm(dim=-1) < radii[:, :, None]+radii[:, None]+.1
        self._place_failed[cliff] |= (clash & pairs & valid[:, :, None] & valid[:, None]).any((1, 2))
