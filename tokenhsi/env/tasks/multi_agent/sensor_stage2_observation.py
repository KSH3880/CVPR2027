import torch

from env.tasks.multi_agent import square_sensor as sensor
from env.tasks.multi_agent import square_sensor_scene as scene
from utils import torch_utils
from utils.sensor_stage2_spec import BRIDGE, CLIFF, FLOOR, ROOM, WAREHOUSE

SENSOR_COLORS = ((.95, .15, .85), (.1, .85, 1.), (1., .7, .05), (.3, 1., .3))
# Hands may brace on walls and feet always carry floor forces, so neither counts as wall contact.
WALL_CONTACT_BODIES = ('pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm',
                       'left_lower_arm', 'right_thigh', 'right_shin', 'left_thigh', 'left_shin')
WALL_CONTACT_TOLERANCE = .03


class SensorStage2Observation:

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
        return sensor.height_observation(root, heading, self._sensor_grid,
            torch.cat((centers, boxes[..., :3]), 1), torch.cat((quats, boxes[..., 3:7]), 1),
            torch.cat((halves, sizes/2), 1), self._env_origins[ids, 2], spec['clip'])

    def _compute_clean_scene_obs(self, env_ids=None):
        self._terrain_ids = env_ids
        obs = super()._compute_clean_scene_obs(env_ids)
        self._terrain_ids = None
        return torch.cat((obs, self._sensor_observation(env_ids).flatten(1)), -1)

    def _compute_clean_humanoid_nodes(self, body_pos, body_rot, body_vel, body_ang_vel, env_origins):
        # Root height is measured from the surface under each human, as on flat Stage1 ground.
        body_pos = body_pos.clone()
        body_pos[..., 2] -= self._terrain_height(body_pos[..., 0, :2], self._terrain_ids)[..., None]
        return super()._compute_clean_humanoid_nodes(body_pos, body_rot, body_vel, body_ang_vel, env_origins)

    def _terrain_height(self, xy, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        centers, quats, halves = self._static_boxes(ids)
        # Room walls are obstacles rather than support; slots 0 and 1 hold every walkable static surface.
        centers = centers[:, :2].clone()
        centers[(self._scene_ids[ids] == ROOM) | (self._scene_ids[ids] == WAREHOUSE), 1, 2] = scene.HIDDEN
        bridge = self._box_states[ids, self._bridge_slot[ids]]
        return sensor.vertical_height(xy, torch.cat((centers, bridge[:, None, :3]), 1),
            torch.cat((quats[:, :2], bridge[:, None, 3:7]), 1),
            torch.cat((halves[:, :2], self._box_size[ids, self._bridge_slot[ids]][:, None]/2), 1),
            self._env_origins[ids, 2])

    def _compute_agent_collision_violation(self, root_pos, min_dist):
        agents = super()._compute_agent_collision_violation(root_pos, min_dist)
        walls = scene.compute_wall_cpa_collision_penalty(
            root_pos, self._humanoid_root_states[..., 7:10], self._wall_starts, self._wall_ends, min_dist,
            self._scene_cfg['room']['wall_ttc_discount'], self.dt)
        agents = torch.maximum(agents, walls*self._wall_collision_scale*(self._scene_ids == ROOM)[:, None])
        if self._warehouse:
            from env.tasks.multi_agent.warehouse_sensor import collision
            agents = collision(self, agents, root_pos, min_dist)
        return agents

    def _compute_reset(self):
        ground = self._terrain_height(self._rigid_body_pos[..., :2])
        fallen = scene.terrain_fall(self._contact_forces, self._rigid_body_pos, self._contact_body_ids, ground,
                                    self._termination_heights, self.progress_buf, self._env_origins[:, 2],
                                    self._start_surface == FLOOR)
        if not self._enable_early_termination:
            fallen.zero_()
        ended = fallen | self._wall_contact_step()
        if self._warehouse:
            from env.tasks.multi_agent.warehouse_sensor import push_box_contact_step
            ended |= push_box_contact_step(self)
        self._terminate_buf[:] = ended.flatten().long()
        done = self.progress_buf >= self.max_episode_length
        formal = getattr(self.cfg['args'], 'eval_protocol', '')
        self.reset_buf[:] = (done if formal else ended.any(-1) | done).long()
        self._record_scene_outcomes(fallen, done)

    def _wall_contact_step(self):
        """Count consecutive Room/Warehouse wall-contact steps; return people over the configured limit."""
        touching = torch.zeros_like(self._wall_contact_steps, dtype=torch.bool)
        ids = ((self._scene_ids == ROOM) | (self._scene_ids == WAREHOUSE)).nonzero().flatten()
        if len(ids):
            centers, quats, halves = self._static_boxes(ids)
            bodies = self._wall_contact_body_ids
            touching[ids] = scene.body_box_contact(self._rigid_body_pos[ids][:, :, bodies],
                self._rigid_body_rot[ids][:, :, bodies], self._contact_forces[ids][:, :, bodies],
                self._wall_contact_shapes, centers[:, 1:], quats[:, 1:], halves[:, 1:], WALL_CONTACT_TOLERANCE)
        touching &= self.progress_buf[:, None] > 1
        steps = torch.where(touching, self._wall_contact_steps+1, torch.zeros_like(self._wall_contact_steps))
        self._wall_contact_steps[:] = steps
        limit = self._wall_contact_limit if self._enable_early_termination else None
        walled = steps >= limit if limit else torch.zeros_like(touching)
        crossed = steps == limit if limit else walled
        for index in range(len(self._scene_names)):
            chosen = self._scene_ids == index
            self._wall_outcomes[index] += torch.stack((touching[chosen].sum(), chosen.sum()*self.num_agents,
                crossed[chosen].sum())).float()
        return walled

    def _record_scene_outcomes(self, fallen, done):
        n = self.num_envs
        rows = torch.arange(n, device=self.device)
        ending = self.reset_buf.bool()
        for index in range(len(self._scene_names)):
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

        if self._warehouse:
            from env.tasks.multi_agent.warehouse_sensor import record
            record(self, ending)

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        if self._bridge_curriculum:
            result.update(self._consume_crossing_curriculum())
        for index, name in enumerate(self._scene_names):
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
            wall = self._wall_outcomes[index]
            result['scene/'+name+'/wall_contact_fraction'] = wall[0]/wall[1].clamp_min(1)
            result['scene/'+name+'/wall_terminations_per_episode'] = wall[2]/outcome[0].clamp_min(1)
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
        if self._warehouse:
            episodes,opened,entered,arrived = self._warehouse_outcomes
            result['warehouse/episodes'] = episodes.clone()
            result['warehouse/opened_fraction'] = opened/episodes.clamp_min(1)
            result['warehouse/worker_episodes'] = self._warehouse_worker_episodes.clone()
            result['warehouse/entered_people_fraction'] = entered/self._warehouse_worker_episodes.clamp_min(1)
            result['warehouse/pusher_arrived_fraction'] = arrived/episodes.clamp_min(1)
            push_box = self._push_box_outcomes
            result['warehouse/push_box_contact_fraction'] = push_box[0]/push_box[1].clamp_min(1)
            result['warehouse/push_box_terminations_per_episode'] = push_box[2]/episodes.clamp_min(1)
            push_box.zero_()
            if self._warehouse_start:
                for index,name in enumerate(('closed','partial','open')):
                    row = self._warehouse_start_outcomes[index]
                    prefix = 'warehouse/start/'+name+'/'
                    result[prefix+'sampled'] = self._warehouse_start_samples[index].clone()
                    result[prefix+'episodes'] = row[0].clone()
                    result[prefix+'opened_fraction'] = row[1]/row[0].clamp_min(1)
                    result[prefix+'worker_episodes'] = self._warehouse_start_worker_episodes[index].clone()
                    result[prefix+'entered_people_fraction'] = row[2]/self._warehouse_start_worker_episodes[index].clamp_min(1)
                    result[prefix+'pusher_arrived_fraction'] = row[3]/row[0].clamp_min(1)
                self._warehouse_start_samples.zero_()
                self._warehouse_start_outcomes.zero_()
                self._warehouse_start_worker_episodes.zero_()
            if self._warehouse_coop:
                from env.tasks.multi_agent.warehouse_coop import diagnostics
                diagnostics(self,result)
            if self._warehouse_before_probability:
                from env.tasks.multi_agent import warehouse_before
                warehouse_before.diagnostics(self,result)
            if self._warehouse_done_probability:
                from env.tasks.multi_agent import warehouse_done
                warehouse_done.diagnostics(self,result)
            if self._warehouse_layout_cache is not None:
                self._warehouse_layout_cache.diagnostics(self,result)
            if self._warehouse_region_matching:
                checks = self._warehouse_region_checks
                result['warehouse/push_region/checked'] = checks[0].clone()
                result['warehouse/push_region/mismatch_fraction'] = checks[1]/checks[0].clamp_min(1)
                checks.zero_()
            self._warehouse_worker_episodes.zero_()
            self._warehouse_outcomes.zero_()
        self._bridge_contact_counts.zero_()
        self._bridge_count_outcomes.zero_()
        self._scene_sampling.zero_()
        self._scene_outcomes.zero_()
        self._wall_outcomes.zero_()
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
                torch.cat((quats, boxes[..., 3:7]), 1), torch.cat((halves, sizes/2), 1), self._env_origins[ids, 2])
            points.append(torch.cat((xy, surface[..., None]), -1)[0])
        sensor.draw_square_sensor(self.gym, self.viewer, self.envs[0], torch.stack(points),
                                  self._scene_cfg['sensor']['resolution'], SENSOR_COLORS)
