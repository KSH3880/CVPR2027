import math

from isaacgym.torch_utils import quat_from_angle_axis, quat_mul, quat_rotate
import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent import square_sensor_scene as scene
from utils import torch_utils
from utils.before_layout import transform
from utils.before_spec import task_objects
from utils.geometry import rotate
from utils.sensor_stage2_spec import BRIDGE, CLIFF, FLOOR, PUSH, ROOM, WAREHOUSE, SURFACE_A, paired_people


class SensorStage2Placement:

    def _group_entities(self, ids):
        """Person, primary box, support box and AT goal of every person, relative to the scene origin."""
        k, m = len(ids), self.num_agents
        rows = torch.arange(k, device=self.device)[:, None]
        people = torch.arange(m, device=self.device)[None].expand(k, -1)
        templates, predecessor = self._templates[ids], self._before_predecessor[ids]
        source, support = self._agent_box_assignment[ids], 3*m+people
        valid_boxes = task_objects(predecessor, templates)
        origin = self._env_origins[ids, None, :2]
        boxes, sizes = self._box_states[ids, :, :2], self._box_size[ids, :, :2].norm(dim=-1)/2
        source_radius, support_radius = sizes[rows, source], sizes[rows, support]
        cfg = self._scene_cfg
        xy = torch.stack((self._humanoid_root_states[ids, :, :2]-origin, boxes[rows, source]-origin,
                          boxes[rows, support]-origin, self._tar_pos[ids, :, :2]-origin), 2)
        human_radius = torch.full_like(source_radius, scene.HUMAN_RADIUS)
        radius = torch.stack((human_radius, source_radius, support_radius, source_radius), 2)
        margin = radius+cfg['edge_margin']
        margin[..., 0] = cfg['human_margin']
        valid = torch.stack((torch.ones_like(valid_boxes[:, :m]), valid_boxes[:, :m], valid_boxes[:, m:],
                             templates == 3), 2)
        human = torch.zeros_like(valid)
        human[..., 0] = True
        return dict(xy=xy, radius=radius, margin=margin, valid=valid, human=human, source=source, support=support)

    def _place_scene(self, ids, room_retry=False):
        warehouse = self._scene_ids[ids] == WAREHOUSE
        if self._warehouse_before_probability:
            flagged = warehouse & self._warehouse_before_pairs[ids]
            if flagged.any():
                from env.tasks.multi_agent.warehouse_before import place as place_apron
                place_apron(self, ids[flagged])
            ids, warehouse = ids[~flagged], warehouse[~flagged]
        if warehouse.any():
            from env.tasks.multi_agent.warehouse_sensor import place
            place(self, ids[warehouse])
            if self._warehouse_done_probability:
                from env.tasks.multi_agent import warehouse_done
                warehouse_done.place(self, ids[warehouse])
        ids = ids[~warehouse]
        if not len(ids):
            return
        k, m = len(ids), self.num_agents
        self._place_failed[ids] = False
        templates = self._templates[ids]
        predecessor = self._before_predecessor[ids]
        people = torch.arange(m, device=self.device)[None].expand(k, -1)
        head = torch.where(predecessor >= 0, predecessor, people)
        paired = paired_people(predecessor)
        bridge = self._scene_ids[ids] == BRIDGE
        skip = self._unreachable[ids] | (self._surface[ids] == FLOOR)
        skip |= self._bridge_pushers[ids] | (bridge[:, None] & (people == self._roles[ids, 1:2]))
        if self._bridge_curriculum:
            skip |= self._crosser_mask[ids]
        entities = self._group_entities(ids)
        occupied = scene.Occupancy(k, 5*m, self.device)
        rows = bridge.nonzero().flatten()
        if len(rows):
            if self._bridge_curriculum:
                for walk in (False, True):
                    selected = rows[self._bridge_walk[ids[rows]] == walk]
                    if len(selected):
                        self._place_bridge(ids, selected, occupied)
            else:
                self._place_bridge(ids, rows, occupied)
        leaders = range(m)
        room_retry = room_retry and bool((self._scene_ids[ids] == ROOM).all())
        if room_retry:
            leaders = torch.randperm(m, device=self.device).tolist()
        for pair_pass in (True, False):
            for leader in leaders:
                chosen = (head[:, leader] == leader) & ~skip[:, leader] & (paired[:, leader] == pair_pass)
                rows = chosen.nonzero().flatten()
                if len(rows):
                    self._place_group(ids, rows, leader, head, entities, occupied, 4 if room_retry else 1)
        self._shift_floor_people(ids)
        for person in range(m):
            rows = self._unreachable[ids, person].nonzero().flatten()
            if len(rows):
                self._place_unreachable(ids, rows, person, entities, occupied)
        self._lift_feet(ids)

    def _rigid(self, states, q, shift, origin):
        local = states.clone()
        local[..., :3] -= origin
        result = transform(local, q, shift)
        result[..., :3] += origin
        return result

    def _place_group(self, ids, rows, leader, head, entities, occupied, retry_scale=1, region=None):
        env = ids[rows]
        m = self.num_agents
        current = self._scene_ids[env]
        surface = self._surface[env, leader]
        if region is None:
            center = self._surface_center[env, current, surface-1]
            half = self._surface_half[env, current, surface-1]
        else:
            center, half = region[0][rows], region[1][rows]
        member = head[rows] == leader
        pick = lambda key: entities[key][rows].flatten(1, 2)
        xy, radius, margin, human = pick('xy'), pick('radius'), pick('margin'), pick('human')
        valid = (entities['valid'][rows] & member[..., None]).flatten(1)
        room = current == ROOM
        bridge = current == BRIDGE
        room_cfg, bridge_cfg = self._scene_cfg['room'], self._scene_cfg['bridge']
        partition_y = self._room_half[env, 1]-room_cfg['passage_width']
        at_person = self._templates[env, leader] == 3
        wants_cross = room & at_person & (torch.rand(len(env), device=self.device) < room_cfg['pickup_cross_probability'])
        tries = retry_scale*self._scene_cfg['placement_tries']
        person_index, box_index = 4*leader, 4*leader+1

        def blocked_path(candidate, chosen):
            start, end = candidate[:, person_index], candidate[:, box_index]
            opposite = start[:, 0]*end[:, 0] < 0
            crossing = start[:, 1]+(end[:, 1]-start[:, 1])*(-start[:, 0])/(end[:, 0]-start[:, 0]).masked_fill(~opposite, 1.)
            return opposite & (crossing.abs() < partition_y[chosen])

        def accept(candidate, pending, attempt):
            good = ~occupied.conflicts(rows[pending], candidate, radius[pending], valid[pending], surface[pending])
            in_room = room[pending]
            if in_room.any():
                reach = radius[pending]+room_cfg['wall_thickness']/2
                hit = (candidate[..., 0].abs() < reach) & (candidate[..., 1].abs() < partition_y[pending, None]+radius[pending])
                good &= ~(in_room & (hit & valid[pending]).any(-1))
                if attempt < tries//2:
                    good &= ~wants_cross[pending] | blocked_path(candidate, pending)
            in_bridge = bridge[pending]
            if in_bridge.any():
                keepout = self._bridge_gap[env[pending], None]/2+bridge_cfg['landing_box_clearance']+radius[pending]
                near = (candidate-self._bridge_landing[env[pending], None]).norm(dim=-1) < keepout
                good &= ~(in_bridge & (near & valid[pending] & ~human[pending]).any(-1))
            return good

        sampler = scene.place_rigid if region is None else scene.place_rigid_batched
        angle, offset, placed = sampler(xy, margin, valid, center, half, accept, tries)
        self._place_failed[env[~placed]] = True
        final = scene.rotate_xy(xy, angle[:, None])+offset[:, None]
        if room.any():
            blocked = blocked_path(final, torch.arange(len(env), device=self.device))
            self._scene_counters[2] += (wants_cross & placed).sum()
            self._scene_counters[3] += (room & at_person & placed & blocked).sum()
            self._scene_counters[4] += (room & at_person & placed).sum()
        chosen = placed.nonzero().flatten()
        if not len(chosen):
            return
        occupied.add(rows[chosen], final[chosen], radius[chosen], valid[chosen], surface[chosen])
        structure_yaw = self._structure_yaw[env[chosen], current[chosen]]
        q = scene.yaw_quat(structure_yaw+angle[chosen])
        shift_xy = scene.rotate_xy(offset[chosen], structure_yaw)+self._structure_shift[env[chosen], current[chosen]]
        lift = self._surface_height[env[chosen], current[chosen]]
        shift = torch.cat((shift_xy, lift[:, None]), -1)
        for person in range(m):
            selected = member[chosen, person].nonzero().flatten()
            if not len(selected):
                continue
            e, qp, sp = env[chosen[selected]], q[selected], shift[selected]
            origin = self._env_origins[e]
            self._humanoid_root_states[e, person] = self._rigid(self._humanoid_root_states[e, person], qp, sp, origin)
            self._kinematic_humanoid_rigid_body_states[e, person] = self._rigid(
                self._kinematic_humanoid_rigid_body_states[e, person], qp, sp, origin[:, None])
            goal = self._tar_pos[e, person]-origin
            self._tar_pos[e, person] = rotate(qp, goal)+sp+origin
            local_rows = rows[chosen[selected]]
            for column, key in ((1, 'source'), (2, 'support')):
                has_box = entities['valid'][local_rows, person, column]
                box_rows, box = e[has_box], entities[key][local_rows[has_box], person]
                if len(box_rows):
                    self._box_states[box_rows, box] = self._rigid(
                        self._box_states[box_rows, box], qp[has_box], sp[has_box], origin[has_box])

    def _place_person(self, env, person, xy, yaw, lift):
        count, nb = len(env), self.num_bodies
        root = self._humanoid_root_states[env, person].clone()
        body = self._kinematic_humanoid_rigid_body_states[env, person].clone()
        old = torch_utils.calc_heading(root[:, 3:7])
        axis = torch.tensor([0., 0., 1.], device=self.device).expand(count, -1)
        q = quat_from_angle_axis(yaw-old, axis)
        qb = q[:, None].expand(-1, nb, -1).reshape(-1, 4)
        offset = body[..., :3]-root[:, None, :3]
        root[:, :2] = xy
        root[:, 2] += lift
        root[:, 3:7] = quat_mul(q, root[:, 3:7])
        for start in (7, 10):
            root[:, start:start+3] = quat_rotate(q, root[:, start:start+3])
            body[..., start:start+3] = quat_rotate(qb, body[..., start:start+3].reshape(-1, 3)).view(count, nb, 3)
        body[..., :3] = quat_rotate(qb, offset.reshape(-1, 3)).view_as(offset)+root[:, None, :3]
        body[..., 3:7] = quat_mul(qb, body[..., 3:7].reshape(-1, 4)).view(count, nb, 4)
        moved = root[:, :2]-self._humanoid_root_states[env, person, :2]
        self._humanoid_root_states[env, person] = root
        self._kinematic_humanoid_rigid_body_states[env, person] = body
        return moved

    def _heading_to(self, start, target):
        direction = target-start
        jitter = (torch.rand(len(start), device=self.device)*2-1)*self._scene_cfg['heading_jitter']
        return torch.atan2(direction[:, 1], direction[:, 0])+jitter

    def _move_neutral_goal(self, env, person, moved, lift):
        person = torch.as_tensor(person, device=self.device).expand(len(env))
        neutral = self._templates[env, person] != 3
        e, owner = env[neutral], person[neutral]
        self._tar_pos[e, owner, :2] += moved[neutral]
        self._tar_pos[e, owner, 2] += lift[neutral]

    def _place_bridge(self, ids, rows, occupied):
        cfg = self._scene_cfg['bridge']
        env = ids[rows]
        m, slot = self.num_agents, self._bridge_slot[env]
        count = len(env)
        origin = self._env_origins[env]
        yaw, shift = self._structure_yaw[env, BRIDGE], self._structure_shift[env, BRIDGE]
        world = lambda xy: scene.to_world(xy, yaw, shift, origin)
        half_a, half_b = self._bridge_half_a[env], self._bridge_half_b[env]
        far, gap, bridge = self._bridge_far[env], self._bridge_gap[env], self._box_size[env, slot]
        layout = scene.bridge_layout(half_a, half_b, far, gap, bridge, cfg, self._bridge_near[env], self._bridge_side[env],
                                    training_mode=self._bridge_start_mode[env] if self._bridge_training_init else None)
        self._bridge_landing[env] = layout['landing']
        height = self._surface_height[env, BRIDGE]

        box = torch.zeros(count, 13, device=self.device)
        box[:, :2] = world(layout['start'])
        box[:, 2] = origin[:, 2]+bridge[:, 2]/2
        box[:, 3:7] = scene.yaw_quat(yaw+torch.where(layout['sign'] > 0, math.pi, 0.))
        self._box_states[env, slot] = box
        landing = torch.cat((world(layout['landing']), box[:, 2:3]), -1)
        self._bridge_landing_world[env] = landing
        pusher = self._roles[env, 0]
        pusher_xy = world(layout['pusher'])
        heading = self._heading_to(pusher_xy, box[:, :2])
        moved = self._place_person(env, pusher, pusher_xy, heading, torch.zeros(count, device=self.device))
        self._push_goals[env, pusher] = landing
        self._tar_pos[env, pusher] = landing

        for person in range(m):
            chosen = self._bridge_pushers[env,person]
            es = env[chosen]
            if not len(es):
                continue
            local = layout['pusher'][chosen].clone()
            local[:,0] += bridge[chosen,0]/self._bridge_count[es]*(self._regions[es,person]+.5)-bridge[chosen,0]/2
            xy = world(layout['pusher']).clone()
            xy[chosen] = scene.to_world(local,yaw[chosen],shift[chosen],origin[chosen])
            self._place_person(es,person,xy[chosen],self._heading_to(xy[chosen],box[chosen,:2]),torch.zeros(len(es),device=self.device))
            self._tar_pos[es,person] = landing[chosen]

        if self._bridge_curriculum and self._bridge_walk[env].all():
            self._place_curriculum_walk(env, rows, occupied)
            return
        crosser = self._roles[env, 1]
        task = self._templates[env, crosser]
        carry_from_start = self._bridge_carry_from_start[env]
        dest, start = self._surface[env, crosser]-1, self._start_surface[env, crosser]-1
        centers, halves = self._surface_center[env, BRIDGE], self._surface_half[env, BRIDGE]
        index = torch.arange(count, device=self.device)
        source = self._agent_box_assignment[env,crosser]
        support = 3*m+crosser
        sizes = self._box_size[env]
        source_radius = sizes[index, source, :2].norm(dim=-1)/2
        second_radius = torch.where(task == 4, sizes[index, support, :2].norm(dim=-1)/2, source_radius)
        payload, second, human, failed = scene.crosser_layout(
            task, centers[index, dest], halves[index, dest], centers[index, start], halves[index, start],
            source_radius, second_radius, layout['landing'], bridge, gap, self._bridge_aligned[env],
            self._scene_cfg['edge_margin'], self._scene_cfg['human_margin'], cfg, carry_from_start,
            present=(self._bridge_start_mode[env] == 0) if self._bridge_training_init else None)
        self._place_failed[env[failed]] = True

        stage_z = origin[:, 2]+height
        self._box_states[env, source, :2] = world(payload)
        self._box_states[env, source, 2] = stage_z+sizes[index, source, 2]/2
        goal_rows = task == 3
        self._tar_pos[env[goal_rows], crosser[goal_rows], :2] = world(second)[goal_rows]
        self._tar_pos[env[goal_rows], crosser[goal_rows], 2] = (stage_z+sizes[index, source, 2]/2)[goal_rows]
        top = task == 4
        self._box_states[env[top], support[top], :2] = world(second)[top]
        self._box_states[env[top], support[top], 2] = (stage_z+sizes[index, support, 2]/2)[top]
        crosser_xy, payload_xy = world(human), world(payload)
        moved = self._place_person(env, crosser, crosser_xy, self._heading_to(crosser_xy, payload_xy), height)
        self._move_neutral_goal(env, crosser, moved, height)

        has_second = (task == 3) | (task == 4)
        xy = torch.stack((human, payload, second), 1)
        radius = torch.stack((torch.full_like(source_radius, scene.HUMAN_RADIUS), source_radius, second_radius), 1)
        ok = ~failed
        start_valid = torch.stack((torch.ones_like(has_second), carry_from_start), 1)
        occupied.add(rows[ok], xy[ok, :2], radius[ok, :2], start_valid[ok], start[ok]+1)
        valid = torch.stack((torch.zeros_like(has_second), ~carry_from_start, has_second), 1)
        occupied.add(rows[ok], xy[ok], radius[ok], valid[ok], dest[ok]+1)

    def _place_unreachable(self, ids, rows, person, entities, occupied):
        cfg, cliff = self._scene_cfg, self._scene_cfg['cliff']
        env = ids[rows]
        count = len(env)
        index = torch.arange(count, device=self.device)
        yaw, shift = self._structure_yaw[env, CLIFF], self._structure_shift[env, CLIFF]
        origin = self._env_origins[env]
        half = self._surface_half[env, CLIFF, 0]
        source = entities['source'][rows, person]
        support = entities['support'][rows, person]
        radius = entities['radius'][rows, person, 1]
        task = self._templates[env, person]
        second_at = (self._tar_pos[env, person, :2]-self._box_states[env, source, :2]).norm(dim=-1)
        second_top = (self._box_states[env, support, :2]-self._box_states[env, source, :2]).norm(dim=-1)
        second_distance = torch.where(task == 4, second_top, second_at)
        human = torch.zeros(count, 2, device=self.device)
        payload, second = torch.zeros_like(human), torch.zeros_like(human)
        pending = torch.ones(count, dtype=torch.bool, device=self.device)
        one = torch.ones(count, 1, dtype=torch.bool, device=self.device)
        person_radius = torch.full((count, 1), scene.HUMAN_RADIUS, device=self.device)
        surface = torch.full((count,), SURFACE_A, device=self.device)
        for _ in range(cfg['placement_tries']):
            trial = scene.offstage_layout(half, radius, second_distance, cliff['object_gap'], cfg['human_margin'],
                                          cliff['max_source_distance'])
            clash = occupied.conflicts(rows, trial[0][:, None], person_radius, one, surface)
            keep = pending & ~clash
            human[keep], payload[keep], second[keep] = trial[0][keep], trial[1][keep], trial[2][keep]
            pending &= clash
            if not pending.any():
                break
        self._place_failed[env[pending]] = True
        world = lambda xy: scene.to_world(xy, yaw, shift, origin)
        human_xy, payload_xy = world(human), world(payload)
        self._box_states[env, source, :2] = payload_xy
        goal_rows = task == 3
        self._tar_pos[env[goal_rows], person, :2] = world(second)[goal_rows]
        top = task == 4
        self._box_states[env[top], support[top], :2] = world(second)[top]
        lift = self._surface_height[env, CLIFF]
        moved = self._place_person(env, person, human_xy, self._heading_to(human_xy, payload_xy), lift)
        self._move_neutral_goal(env, person, moved, lift)
        ok = ~pending
        occupied.add(rows[ok], human[ok, None], person_radius[ok], one[ok], surface[ok])

    def _lift_feet(self, ids):
        raised = self._start_surface[ids] > FLOOR
        if not raised.any():
            return
        body = self._kinematic_humanoid_rigid_body_states[ids]
        lowest = sensor.lowest_foot_height(body, self._foot_body_ids, self._foot_vertices)
        support = (self._env_origins[ids, 2]+self._surface_height[ids, self._scene_ids[ids]])[:, None]
        if self._bridge_curriculum:
            on_bridge = self._bridge_walk[ids, None] & (self._cross_phase[ids] > 0)
            top = self._env_origins[ids, 2]+self._box_size[ids, self._bridge_slot[ids], 2]
            support = torch.where(on_bridge, top[:, None], support)
        lift = ((support+self._scene_cfg['foot_clearance']-lowest).clamp_min(0.))*raised
        self._humanoid_root_states[ids, :, 2] += lift
        self._kinematic_humanoid_rigid_body_states[ids, :, :, 2] += lift[:, :, None]

    def _active_physical(self, ids):
        active = torch.zeros(len(ids),self.num_physical_objects,dtype=torch.bool,device=self.device)
        active.scatter_(1,self._logical_box_order[ids],True)
        return active

    def _shift_floor_people(self, ids):
        """Slide all floor tasks and distractors together off the structure, keeping their relative layout."""
        k, m = len(ids), self.num_agents
        rows = torch.arange(k, device=self.device)
        people = torch.arange(m, device=self.device)[None].expand(k, -1)
        current = self._scene_ids[ids]
        templates = self._templates[ids]
        pusher = self._bridge_pushers[ids]
        floor = (self._start_surface[ids] == FLOOR) & ~pusher
        primary = self._agent_box_assignment[ids]
        moved_boxes = self._active_physical(ids)
        moved_boxes[rows[:, None], primary] &= floor
        moved_boxes[rows[:, None], 3*m+people] &= floor
        moved_boxes[rows, self._bridge_slot[ids]] = False
        origin = self._env_origins[ids, None, :2]
        axis, direction = self._structure_axis[ids, current], self._structure_direction[ids, current]
        sizes = self._box_size[ids, :, :2].norm(dim=-1)/2
        goals = self._tar_pos[ids, :, :2]-origin
        boxes = self._box_states[ids, :, :2]-origin
        own_radius = sizes[rows[:, None], primary]
        coordinate = torch.cat((self._humanoid_root_states[ids, :, :2]-origin, boxes, goals), 1)
        radius = torch.cat((torch.full((k, m), scene.HUMAN_RADIUS, device=self.device), sizes, own_radius), 1)
        counted = torch.cat((floor, moved_boxes, floor & ((templates == 3) | (templates == PUSH))), 1)
        along = coordinate[rows, :, axis]*direction[:, None]+radius
        extreme = along.masked_fill(~counted, -torch.inf).amax(-1)
        limit = self._structure_near[ids, current]-self._scene_cfg['floor_margin']
        distance = (extreme-limit).clamp_min(0.)
        shift = torch.zeros(k, 2, device=self.device)
        shift[rows, axis] = -direction*distance
        chosen = rows[distance > 0]
        for person in range(m):
            selected = chosen[floor[chosen, person]]
            env = ids[selected]
            self._humanoid_root_states[env, person, :2] += shift[selected]
            self._kinematic_humanoid_rigid_body_states[env, person, :, :2] += shift[selected, None]
            self._tar_pos[env, person, :2] += shift[selected]
            self._push_goals[env, person, :2] += shift[selected]
        row, box = moved_boxes[chosen].nonzero(as_tuple=True)
        self._box_states[ids[chosen[row]], box, :2] += shift[chosen[row]]
