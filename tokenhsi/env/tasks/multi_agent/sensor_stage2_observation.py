import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent import square_sensor_scene as scene
from utils import torch_utils
from utils.sensor_stage2_spec import BRIDGE, CLIFF, FLOOR, ROOM, SCENES

SENSOR_COLORS = ((.95, .15, .85), (.1, .85, 1.), (1., .7, .05), (.3, 1., .3))


class SensorStage2Observation:

    def _height_chamfer(self, ids, count):
        if not self._bridge_entry_chamfer:
            return None
        from env.tasks.multi_agent.bridge_entry_chamfer import WIDTH, DEPTH
        result = torch.zeros(len(ids), count, 3, device=self.device)
        bridge = self._scene_ids[ids] == BRIDGE
        result[bridge, 0, 0], result[bridge, 1, 0] = 1., -1.
        result[bridge, :2, 1], result[bridge, :2, 2] = WIDTH, DEPTH
        return result

    def _static_boxes(self, ids):
        current = self._scene_ids[ids]
        return self._static_centers[ids, current], self._static_quat[ids, current], self._static_half[ids, current]

    def _sensor_observation(self, env_ids=None):
        spec = self._scene_cfg['sensor']
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if env_ids is None:
            root, rotation = self._rigid_body_pos[:, :, 0], self._rigid_body_rot[:, :, 0]
        else:
            body = self._kinematic_humanoid_rigid_body_states[env_ids]
            root, rotation = body[:, :, 0, :3], body[:, :, 0, 3:7]
        heading = torch_utils.calc_heading(rotation.reshape(-1, 4)).view(rotation.shape[:2])
        centers, quats, halves = self._static_boxes(ids)
        boxes = self._gather_entity_slots(self._box_states[ids],self._logical_box_order[ids])
        sizes = self._gather_entity_slots(self._box_size[ids],self._logical_box_order[ids])
        exclude = None
        if spec.get('exclude_own_payload', False):
            templates, pre = self._templates[ids], self._before_predecessor[ids]
            owner = ((templates == 3) | (templates == 4)) & ((pre < 0) | (templates == 4))
            own = (self._logical_box_order[ids][:, None] == self._agent_box_assignment[ids][..., None]) & owner[..., None]
            exclude = torch.cat((torch.zeros_like(own[..., :1]).expand(-1, -1, centers.shape[1]), own), -1)
        return sensor.height_observation(root, heading, self._sensor_grid,
            torch.cat((centers, boxes[..., :3]), 1), torch.cat((quats, boxes[..., 3:7]), 1),
            torch.cat((halves, sizes/2), 1), self._env_origins[ids, 2], spec['clip'],
            self._height_chamfer(ids, centers.shape[1]+boxes.shape[1]), exclude)

    def _compute_clean_scene_obs(self, env_ids=None):
        self._terrain_ids = env_ids
        obs = super()._compute_clean_scene_obs(env_ids)
        self._terrain_ids = None
        return torch.cat((obs, self._sensor_observation(env_ids).flatten(1)), -1)

    def _compute_clean_humanoid_nodes(self, body_pos, body_rot, body_vel, body_ang_vel, env_origins):
        body_pos = body_pos.clone()
        height = self._terrain_height(body_pos[..., 0, :2], self._terrain_ids)
        if self._scene_cfg['bridge'].get('humanoid_height_reference') == 'fixed_start_surface':
            ids = slice(None) if self._terrain_ids is None else self._terrain_ids
            fixed = self._env_origins[ids, 2, None]+(
                self._start_surface[ids] > FLOOR)*self._surface_height[ids, BRIDGE, None]
            height = torch.where((self._scene_ids[ids] == BRIDGE)[:, None], fixed, height)
        body_pos[..., 2] -= height[..., None]
        return super()._compute_clean_humanoid_nodes(body_pos, body_rot, body_vel, body_ang_vel, env_origins)

    def _terrain_height(self, xy, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        centers, quats, halves = self._static_boxes(ids)
        # Room walls are obstacles rather than support; slots 0 and 1 hold every walkable static surface.
        centers = centers[:, :2].clone()
        centers[self._scene_ids[ids] == ROOM, 1, 2] = scene.HIDDEN
        bridge = self._box_states[ids, self._bridge_slot[ids]]
        return sensor.vertical_height(xy, torch.cat((centers, bridge[:, None, :3]), 1),
            torch.cat((quats[:, :2], bridge[:, None, 3:7]), 1),
            torch.cat((halves[:, :2], self._box_size[ids, self._bridge_slot[ids]][:, None]/2), 1),
            self._env_origins[ids, 2], self._height_chamfer(ids, 3))

    def _compute_agent_collision_violation(self, root_pos, min_dist):
        agents = super()._compute_agent_collision_violation(root_pos, min_dist)
        walls = scene.compute_wall_cpa_collision_penalty(
            root_pos, self._humanoid_root_states[..., 7:10], self._wall_starts, self._wall_ends, min_dist,
            self._scene_cfg['room']['wall_ttc_discount'], self.dt)
        return torch.maximum(agents, walls*(self._scene_ids == ROOM)[:, None])

    def _compute_reset(self):
        ground = self._terrain_height(self._rigid_body_pos[..., :2])
        fallen = scene.terrain_fall(self._contact_forces, self._rigid_body_pos, self._contact_body_ids, ground,
                                    self._termination_heights, self.progress_buf, self._env_origins[:, 2],
                                    self._start_surface == FLOOR)
        if not self._enable_early_termination:
            fallen.zero_()
        self._terminate_buf[:] = fallen.flatten().long()
        done = self.progress_buf >= self.max_episode_length
        formal = getattr(self.cfg['args'], 'eval_protocol', '')
        self.reset_buf[:] = (done if formal else fallen.any(-1) | done).long()
        self._record_scene_outcomes(fallen, done)

    def _record_scene_outcomes(self, fallen, done):
        n = self.num_envs
        rows = torch.arange(n, device=self.device)
        ending = self.reset_buf.bool()
        for index in range(len(SCENES)):
            chosen = self._scene_ids == index
            self._scene_outcomes[index] += torch.stack((
                (chosen & ending).sum(), chosen.sum(), fallen[chosen].sum(), (chosen & done).sum())).float()
        roots = self._rigid_body_pos[:, :, 0]
        on_cliff = (self._scene_ids == CLIFF)[:, None] & (self._start_surface > FLOOR)
        if on_cliff.any():
            origin = self._env_origins[:, None, :2]
            shift = self._structure_shift[:, CLIFF, None]
            local = scene.rotate_xy(roots[..., :2]-origin-shift, -self._structure_yaw[:, CLIFF, None])
            clearance = (self._surface_half[:, CLIFF, 0, None]-local.abs()).amin(-1)
            near = on_cliff & (clearance >= 0) & (clearance < 1.)
            speed = self._rigid_body_vel[:, :, 0, :2].norm(dim=-1)
            self._scene_counters[0] += near.sum()
            self._scene_counters[1] += (near & (speed < .15)).sum()
        bridge = self._scene_ids == BRIDGE
        distance = (self._box_states[rows, self._bridge_slot, :2]-self._bridge_landing_world[:, :2]).norm(dim=-1)
        self._bridge_arrived |= bridge & (distance < self._relation_cfg['at']['success_distance'])
        crosser = self._roles[:, 1]
        root = roots[rows, crosser]
        stage = (self._surface[rows, crosser]-1).clamp_min(0)
        center = self._surface_center[rows, BRIDGE, stage]
        local = scene.rotate_xy(root[:, :2]-self._env_origins[:, :2]-self._structure_shift[:, BRIDGE],
                                -self._structure_yaw[:, BRIDGE])
        over = ((local-center).abs() <= self._surface_half[rows, BRIDGE, stage]).all(-1)
        high = root[:, 2] > self._env_origins[:, 2]+self._surface_height[:, BRIDGE]+.6
        self._bridge_crossed |= bridge & over & high
        if self._bridge_curriculum:
            self._record_crossing_curriculum(fallen, ending, roots)
        if self._bridge_training_init:
            box = self._box_states[rows, self._bridge_slot]
            size = self._box_size[rows, self._bridge_slot]
            heading = torch_utils.calc_heading(box[:, 3:7])
            xy = scene.rotate_xy(root[:, :2]-box[:, :2], -heading)
            over_box = (xy.abs() <= size[:, :2]/2).all(-1)
            self._bridge_entered |= bridge & over_box & (root[:, 2] > box[:, 2]+size[:, 2]/2+.4) & ~fallen[rows, crosser]
        _,_,_,_,_,contact = self._contact_geometry(rows,self._rigid_body_pos)
        contacted = (contact['on_face'] | ~self._bridge_pushers).all(-1)
        for index in range(3):
            chosen = bridge & (self._bridge_count == index+1)
            self._bridge_contact_counts[index] += torch.stack((chosen.sum(),(chosen & contacted).sum()))
        ended = ending & bridge
        if ended.any():
            mode = self._bridge_near.long()+self._bridge_aligned.long()
            values = torch.stack((torch.ones_like(ended), self._bridge_arrived, self._bridge_crossed,
                                  fallen[rows, crosser]), -1).float()
            for index in range(3):
                self._bridge_outcomes[index] += values[ended & (mode == index)].sum(0)
                self._bridge_count_outcomes[index] += values[ended & (self._bridge_count == index+1)].sum(0)
            if self._bridge_training_init:
                values = torch.stack((torch.ones_like(ended), self._bridge_arrived, self._bridge_entered,
                                      self._bridge_crossed, fallen[rows, crosser]), -1).float()
                for mode in range(3):
                    for count in range(3):
                        chosen = ended & (self._bridge_start_mode == mode) & (self._bridge_count == count+1)
                        self._bridge_start_outcomes[mode, count] += values[chosen].sum(0)

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        if self._bridge_curriculum:
            result.update(self._consume_crossing_curriculum())
        for index, name in enumerate(SCENES):
            sampled, outcome = self._scene_sampling[index], self._scene_outcomes[index]
            episodes = sampled[0].clamp_min(1)
            result['scene/'+name+'/episodes'] = sampled[0].clone()
            result['scene/'+name+'/before_pairs'] = sampled[1]/episodes
            result['scene/'+name+'/solo_people'] = sampled[2]/episodes
            result['scene/'+name+'/solo_push'] = sampled[3]/episodes
            result['scene/'+name+'/unreachable_people'] = sampled[4]/episodes
            result['scene/'+name+'/falls_per_episode'] = outcome[2]/outcome[0].clamp_min(1)
            result['scene/'+name+'/timeout_fraction'] = outcome[3]/outcome[0].clamp_min(1)
            result['scene/'+name+'/episode_steps'] = outcome[1]/outcome[0].clamp_min(1)
        counters = self._scene_counters
        result['cliff/stopped_near_edge_fraction'] = counters[1]/counters[0].clamp_min(1)
        result['cliff/near_edge_person_steps'] = counters[0].clone()
        result['room/cross_requested_fraction'] = counters[2]/counters[4].clamp_min(1)
        result['room/blocked_path_fraction'] = counters[3]/counters[4].clamp_min(1)
        total = self._bridge_outcomes.sum(0)
        for label, row in zip(('original', 'near', 'aligned', 'all'), (*self._bridge_outcomes, total)):
            if self._bridge_training_init and label != 'all':
                continue
            for column, metric in enumerate(('arrived', 'crossed', 'crosser_falls'), 1):
                result['bridge/'+label+'_'+metric] = row[column]/row[0].clamp_min(1)
        if self._bridge_training_init:
            for mode, name in enumerate(('present', 'near', 'insert')):
                for count in range(3):
                    prefix = f'bridge/start/{name}/pushers{count+1}/'
                    row = self._bridge_start_outcomes[mode, count]
                    result[prefix+'sampled'] = self._bridge_start_samples[mode, count].clone()
                    result[prefix+'episodes'] = row[0].clone()
                    for column, metric in enumerate(('arrived', 'entered', 'crossed', 'crosser_falls'), 1):
                        result[prefix+metric] = row[column]/row[0].clamp_min(1)
            required = self._bridge_start_outcomes[1:].sum((0, 1))
            result['bridge/push_required/episodes'] = required[0].clone()
            result['bridge/push_required/arrived'] = required[1]/required[0].clamp_min(1)
            self._bridge_start_outcomes.zero_()
            self._bridge_start_samples.zero_()
        for index,row in enumerate(self._bridge_count_outcomes):
            result[f'bridge/pushers{index+1}/episodes'] = row[0].clone()
            for column,metric in enumerate(('arrived','crossed','crosser_falls'),1):
                result[f'bridge/pushers{index+1}/'+metric] = row[column]/row[0].clamp_min(1)
        for index,row in enumerate(self._bridge_contact_counts):
            result[f'bridge/pushers{index+1}/all_on_face_fraction'] = row[1]/row[0].clamp_min(1)
        self._bridge_contact_counts.zero_()
        self._bridge_count_outcomes.zero_()
        self._scene_sampling.zero_()
        self._scene_outcomes.zero_()
        self._bridge_outcomes.zero_()
        counters.zero_()
        return result

    def _draw_ontop_context(self):
        super()._draw_ontop_context()
        if self.viewer is None or not self.debug_viz:
            return
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        root = self._rigid_body_pos[ids, :, 0]
        heading = torch_utils.calc_heading(self._rigid_body_rot[ids, :, 0].reshape(-1, 4)).view(1, self.num_agents)
        centers, quats, halves = self._static_boxes(ids)
        boxes = self._gather_entity_slots(self._box_states[ids],self._logical_box_order[ids])
        sizes = self._gather_entity_slots(self._box_size[ids],self._logical_box_order[ids])
        points = []
        for person in range(self.num_agents):
            xy = sensor.sensor_positions(root[:, person], heading[:, person], self._sensor_grid)
            surface = sensor.vertical_height(xy, torch.cat((centers, boxes[..., :3]), 1),
                torch.cat((quats, boxes[..., 3:7]), 1), torch.cat((halves, sizes/2), 1), self._env_origins[ids, 2],
                self._height_chamfer(ids, centers.shape[1]+boxes.shape[1]))
            points.append(torch.cat((xy, surface[..., None]), -1)[0])
        sensor.draw_square_sensor(self.gym, self.viewer, self.envs[0], torch.stack(points),
                                  self._scene_cfg['sensor']['resolution'], SENSOR_COLORS)
