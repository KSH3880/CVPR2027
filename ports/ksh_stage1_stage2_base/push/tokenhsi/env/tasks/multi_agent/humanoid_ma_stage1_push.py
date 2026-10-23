"""Native Stage1 (2 humans, 4 objects, GTA/task packets) plus PUSH contact.

The existing HOLDING/AT pair topology is reused only in this separate task.
Native HOLDING evaluators and all original experiment configs remain available.
"""
import math
from isaacgym import gymapi
import torch
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, build_amp_observations
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import select_graph, batched
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.push_point import common_face_xy, common_face_state, common_face_contact, local_points, point_geometry, point_reward_phi, point_reward_contact, midpoint_reward_phi, midpoint_force_contact, midpoint_position_contact
from utils.stage1_push import sample_boxes_and_goals, sample_box_sizes, sample_rsi_rows, load_bank, box_tilt_degrees


class NoSizeCache:
    def __init__(self, task): pass


class PushRuntime(Stage1ContextRuntime):
    def __init__(self, task, *args):
        super().__init__(*args); self.task = task; self.rows = None

    def reset(self, ids, *args, **kwargs):
        self.rows = ids
        try: return super().reset(ids, *args, **kwargs)
        finally: self.rows = None

    def success(self, phi, z_error, feet_error, graph, region_error):
        result = super().success(phi, z_error, feet_error, graph, region_error)
        limit = getattr(self.task, 'cfg', {}).get('env', {}).get('pushTask', {}).get('at_success_max_tilt_deg')
        if limit is not None:
            boxes = self.task._box_states if self.rows is None else self.task._box_states[self.rows]
            src = batched(graph.edge_src, phi.shape[0])
            index = (src - graph.num_agents).clamp(0, graph.num_objects - 1)
            tilt = box_tilt_degrees(boxes[..., 3:7]).gather(1, index)
            result = result & ((batched(graph.edge_relation, phi.shape[0]) != AT) | (tilt <= limit))
        if getattr(self.task, 'cfg', {}).get('env', {}).get('pushTask', {}).get('contact_objective') in ('midpoint_only_v1', 'common_face_xy_v1'):
            contact = phi > .9
            if self.task.cfg['env']['pushTask'].get('require_contact_force', False):
                hits = self.task._push_hits if self.rows is None else self.task._push_hits[self.rows]
                contact = hits.gather(1, graph.edge_owner)
            return torch.where(graph.edge_relation == HOLDING, contact, result) & graph.edge_valid
        contact = self.task._push_hits if self.rows is None else self.task._push_hits[self.rows]
        return torch.where(graph.edge_relation == HOLDING,
                           contact.gather(1, graph.edge_owner), result) & graph.edge_valid


class HumanoidMAStage1Push(HumanoidMACarry):
    _solo_cache_class = NoSizeCache

    def __init__(self, cfg, *args, **kwargs):
        if cfg['env'].get('pushTask') is None:
            raise ValueError('Use the separate stage1_push config')
        super().__init__(cfg, *args, **kwargs)
        self._push_bank = load_bank(cfg['env'])
        if cfg['env']['pushTask'].get('rsi_sampling') == 'native_size_cache' and not getattr(self, '_building_push_rsi_cache', False):
            from utils.push_native_rsi import PushNativeRsiCache
            self._push_native_rsi = PushNativeRsiCache(self)

    def _load_box_asset(self):
        n, o = self.num_envs, self.num_objects
        spec = self.cfg['env']['pushBox']
        self._box_size = self._sample_push_box_sizes()
        self._box_scale = self._box_size/1.3
        self._box_density = (spec['mass']/self._box_size.prod(-1)).reshape(-1)
        assets = {}; self._box_assets = []
        for dimensions in self._box_size.reshape(-1, 3).cpu().tolist():
            size = tuple(dimensions)
            if size not in assets:
                options = gymapi.AssetOptions(); options.density = spec['mass']/math.prod(size)
                options.angular_damping = options.linear_damping = .01
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                asset = self.gym.create_box(self.sim, *size, options)
                props = self.gym.get_asset_rigid_shape_properties(asset)
                for prop in props: prop.friction = spec['friction']; prop.restitution = spec['restitution']
                self.gym.set_asset_rigid_shape_properties(asset, props); assets[size] = asset
            self._box_assets.append(assets[size])

    def _sample_push_box_sizes(self):
        return sample_box_sizes(self.num_envs, self.device, self.cfg['env'])

    def _retarget_push_arms(self, body, rotations, velocities, angular, dof, dof_vel, lib):
        if self.cfg['env']['pushTask'].get('rsi_retarget') == 'none':
            return body, rotations, velocities, angular, dof, dof_vel
        from utils.open_push_rsi import lower_push_arms
        hands = self._key_body_ids[:2].cpu().tolist()
        height = self.cfg['env']['pushTask']['height_m']
        args = (lib._motions[0].skeleton_tree.parent_indices.tolist(), hands, self._dof_body_ids, self._dof_offsets)
        if self.cfg['env']['pushTask'].get('rsi_retarget') != 'preserve_hand_height_difference':
            return lower_push_arms(body, rotations, velocities, angular, dof, dof_vel, *args, height)
        targets = body[:, hands, 2]+(height-body[:, hands, 2].mean(-1))[:, None]
        for i, hand in enumerate(hands):
            body, rotations, velocities, angular, dof, dof_vel = lower_push_arms(
                body, rotations, velocities, angular, dof, dof_vel, args[0], [hand], args[2], args[3], targets[:, i])
        return body, rotations, velocities, angular, dof, dof_vel

    def _init_ontop_context_runtime(self):
        super()._init_ontop_context_runtime()
        graph = self.relation_runtime.graph
        self._push_hits = torch.zeros(self.num_envs, self.num_agents, device=self.device, dtype=torch.bool)
        self._push_rsi_interaction = torch.zeros_like(self._push_hits)
        self._push_rsi_requested = torch.zeros_like(self._push_hits)
        self._push_rsi_samples = {k: torch.zeros(self.num_envs, self.num_agents, device=self.device, dtype=dtype)
                                 for k, dtype in (('motion_id', torch.long), ('time', torch.float32),
                                     ('phase', torch.float32), ('box_progress', torch.float32), ('late_requested', torch.bool))}
        self._push_side = torch.full((self.num_envs, self.num_agents), 2, device=self.device, dtype=torch.long)
        height = torch.full((self.num_envs, self.num_agents), self.cfg['env']['pushTask'].get('height_m', .65), device=self.device)
        self._push_local = local_points(self._push_side, self._box_size[:, :self.num_agents], height)
        self.relation_runtime = PushRuntime(self, self.num_envs, graph, self._relation_cfg, self.device)
        self._push_measurements = {}
        self._push_metric_sums = {}
        self._push_metric_count = 0

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        bodies = self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        hands = bodies[:, :, self._key_body_ids[:2]]
        boxes = self._box_states[ids, :2]
        side, local = self._push_side[ids], self._push_local[ids]
        sizes = self._box_size[ids, :2]
        midpoint = self.cfg['env']['pushTask'].get('contact_objective') == 'midpoint_only_v1'
        point_spec = self.cfg['env']['pushPointReward']
        surface = self.cfg['env']['pushTask'].get('contact_objective') == 'common_face_xy_v1'
        if surface:
            score = common_face_state(hands, boxes, sizes, self._tar_pos[ids])
            if not self.cfg['env']['pushTask'].get('require_contact_force', False):
                self._push_hits[ids] = score > .9
            elif env_ids is not None:
                self._push_hits[ids] = False
        elif midpoint:
            score = midpoint_reward_phi(hands, boxes, local, point_spec)
            if not self.cfg['env']['pushTask'].get('require_contact_force', False):
                self._push_hits[ids] = midpoint_position_contact(hands, boxes, local, point_spec)
            elif env_ids is not None:
                # Reset poses do not carry a measured contact from the preceding episode.
                self._push_hits[ids] = False
        else:
            score = point_reward_phi(hands, boxes, sizes, side, local, point_spec)
        graph = select_graph(self.relation_runtime.graph, ids)
        mask = graph.edge_valid & (graph.edge_relation == HOLDING)
        phi = torch.where(mask, score.gather(1, graph.edge_owner), phi)
        approach_xy = (bodies[:, :, 0, :2]-boxes[..., :2]).norm(dim=-1)
        approach = 1/(1+(approach_xy-.5).clamp_min(0))
        diag['progress'] = torch.where(mask, approach.gather(1, graph.edge_owner), diag['progress'])
        if env_ids is None:
            tilt = box_tilt_degrees(boxes[..., 3:7])
            metrics = dict(contact=self._push_hits.float().mean(), at=self.relation_runtime.done.float().mean(),
                           box_tilt_deg=tilt.mean(), box_tilt_max_deg=tilt.max(),
                           box_upright_2deg=(tilt <= 2.).float().mean(),
                           rsi_fallback=(self._push_rsi_requested & ~self._push_rsi_interaction).float().mean(),
                           goal_distance=(boxes[..., :2]-self._tar_pos[..., :2]).norm(dim=-1).mean())
            if surface:
                distances, _, _ = common_face_xy(hands, boxes, sizes, self._tar_pos[ids])
                metrics['contact_max_xy_distance'] = distances.amax(-1).mean()
            else:
                _, delta, _, _ = point_geometry(hands, boxes, sizes, side, local)
                from utils.push_point import point_height_valid
                height_valid = (delta[..., 2].abs() < point_spec['z_reward_sigma_m'] * math.sqrt(-math.log(.9))
                                if midpoint else point_height_valid(delta, point_spec))
                metrics['midpoint_height_in_band'] = height_valid.float().mean()
            self._push_measurements = {key: value.detach() for key, value in metrics.items()}
        return phi, diag

    def _fetch_conditioned_amp_demo(self, count):
        from utils.unified_training import append_family
        result = torch.empty(count, self.get_num_amp_obs(), device=self.device)
        selected = torch.multinomial(torch.tensor(self.cfg['env']['pushTask']['amp_expert_probabilities'], device=self.device), count, replacement=True)
        for family, skill in enumerate(('loco', 'push')):
            ids = (selected == family).nonzero().flatten()
            if not len(ids): continue
            lib = self._motion_lib[skill]; clips = lib.sample_motions(len(ids))
            history = self.dt*(self._num_amp_obs_steps-1)
            times = lib.sample_time(clips, truncate_time=history)+history
            obs = self.build_amp_obs_demo(clips, times, lib).reshape(len(ids), self._num_amp_obs_steps, -1)
            # Preserve the imported 3-family AMP width; this separate experiment uses the manipulation family.
            result[ids] = append_family(obs, torch.zeros(obs.shape[:2], dtype=torch.long, device=self.device)).flatten(1)
        return result

    def _set_pose(self, rows, owners, body, rotations, velocities, angular, dof, dof_vel):
        self._set_env_state(rows, owners, body[:, 0], rotations[:, 0], dof,
                            velocities[:, 0], angular[:, 0], dof_vel)
        self._kinematic_humanoid_rigid_body_states[rows, owners] = torch.cat((body, rotations, velocities, angular), -1)
        self._every_env_init_dof_pos[rows, owners] = dof

    def _reset_envs(self, ids):
        if not len(ids): return
        n = len(ids); spec = self.cfg['env']['pushTask']
        geometry = self._push_reset_geometry()
        self._reset_default_slots = None
        self._reset_ref_slots = {}; self._reset_ref_motion_ids = {}; self._reset_ref_motion_times = {}
        self._sample_episode_graph(ids)
        states, goals, forward, distances = sample_boxes_and_goals(n, self.device, self._env_origins[ids], self.cfg['env'], self._box_size[ids], epoch=getattr(self, '_push_goal_epoch', 0))
        self._box_states[ids] = states; self._tar_pos[ids] = goals; self._push_hits[ids] = False
        self._platform_pos[ids] = self._platform_default_pos[ids]
        self._tar_platform_pos[ids] = self._tar_platform_default_pos[ids]
        rows = ids.repeat_interleave(2); owners = torch.arange(2, device=self.device).repeat(n)
        lib = self._motion_lib['loco']; clips = lib.sample_motions(len(rows)); times = lib.sample_time_rsi(clips)
        root, rot, dof, vel, ang, dof_vel, _ = lib.get_motion_state(clips, times)
        body, rotations, velocities, angular = lib.get_motion_state_max(clips, times)
        target_yaw = yaw_from_quat(states[:, :2, 3:7]).flatten()+math.pi/2
        angle = target_yaw-yaw_from_quat(rot)
        q = torch.zeros_like(rot); q[:, 2] = (angle/2).sin(); q[:, 3] = (angle/2).cos()
        qs = q[:, None].expand_as(rotations)
        world_rotations = multiply(qs, rotations)
        relative = rotate(qs, body-body[:, :1])
        lo, hi = spec['loco_distance_range_m']
        distance = lo+(hi-lo)*torch.rand(len(rows), device=self.device)
        position = states[:, :2, :3].reshape(-1, 3).clone()
        position[:, :2] -= forward.reshape(-1, 3)[:, :2]*distance[:, None]
        position[:, 2] = root[:, 2]
        if geometry is not None:
            position[:,2]+=geometry.floor_shift(relative+position[:,None],world_rotations)
        # Reject own-box body penetration; never silently alter the configured distance range.
        for _ in range(64):
            world = relative+position[:, None]
            inv = states[:, :2, 3:7].reshape(-1, 4).clone(); inv[:, :3] *= -1
            local = rotate(inv[:, None].expand_as(rotations), world-states[:, :2, :3].reshape(-1, 1, 3))
            invalid = (local.abs() < self._box_size[ids, :2].reshape(-1, 1, 3)/2+.04).all(-1).any(-1)
            if geometry is not None:
                invalid |= geometry.invalid(world,world_rotations,states[:, :2].reshape(-1,13),self._box_size[ids,:2].reshape(-1,3))
            if not invalid.any(): break
            distance[invalid] = lo+(hi-lo)*torch.rand(int(invalid.sum()), device=self.device)
            position[invalid, :2] = states[:, :2, :2].reshape(-1, 2)[invalid]-forward.reshape(-1, 3)[invalid, :2]*distance[invalid, None]
        else: raise RuntimeError('Cannot initialize PUSH locomotion without box penetration')
        self._set_pose(rows, owners, world, world_rotations, rotate(qs, velocities), rotate(qs, angular), dof, dof_vel)
        self._reset_ref_slots['loco'] = (rows, owners)
        self._reset_ref_motion_ids['loco'] = clips; self._reset_ref_motion_times['loco'] = times
        probability = spec['interaction_probability'] if not self._is_eval else sum(p for s, p in zip(self.cfg['env']['eval']['skill'], self.cfg['env']['eval']['skillInitProb']) if s == 'push')
        interaction = torch.rand(len(rows), device=self.device) < probability
        self._push_rsi_requested[ids] = interaction.reshape(n, 2)
        if spec.get('rsi_sampling') == 'native_size_cache':
            interaction &= self._push_native_rsi.available(self._box_size[rows, owners])
        self._push_rsi_interaction[ids] = interaction.reshape(n, 2)
        for key, values in self._push_rsi_samples.items():
            values[ids] = -1 if key == 'motion_id' else 0
        if interaction.any():
            r, a = rows[interaction], owners[interaction]
            if spec.get('rsi_sampling') == 'native_size_cache':
                sample = self._push_native_rsi.sample(self._box_size[r, a])
                travel = distances.flatten()[interaction]
                # Keep the sampled clip/time. Only scene box progress is bounded
                # so the configured initial goal distance cannot start already at AT.
                limit = 1-(spec['minimum_rsi_remaining_m']+.001)/travel
                sample['box_progress'] = torch.minimum(sample['box_progress'], limit)
            else:
                sample = sample_rsi_rows(self._push_bank, self._motion_lib['push'], len(r), spec)
            for key, values in self._push_rsi_samples.items(): values[r, a] = sample[key]
            travel = distances.flatten()[interaction]
            if ((1-sample['box_progress'])*travel < spec['minimum_rsi_remaining_m']).any():
                raise ValueError('Screened RSI pose starts too close to AT; preserve the configured goal/RSI laws')
            if len(r): self._apply_push_rsi(r, a, sample, states, ids, goals)
        if spec.get('height_mode') == 'amp_reference':
            # Loco starts also get a fixed target sampled from the native PUSH reference pool.
            r, a = rows[~interaction], owners[~interaction]
            if len(r):
                ref = self._push_native_rsi.sample(self._box_size[r, a])
                body = self._motion_lib['push'].get_motion_state_max(ref['motion_id'], ref['time'])[0]
                self._set_amp_midpoint_height(r, a, body)
        # Preserve the original four-object scene; only the first two are bound to this PUSH graph.
        self._update_box_assignment_colors(ids)
        self._reset_env_tensors(ids); self._refresh_sim_tensors()
        self._reset_relation_history(ids); self._compute_observations(ids); self._init_amp_obs(ids)

    def _set_amp_midpoint_height(self, rows, owners, body):
        """Freeze the sampled reference height; never follow the policy's current hands."""
        height = body[:, self._key_body_ids[:2], 2].mean(-1)
        size_z = self._box_size[rows, owners, 2]
        if not torch.isfinite(height).all() or ((height <= 0) | (height >= size_z)).any():
            raise ValueError('Sampled AMP midpoint height is outside the PUSH box face')
        self._push_local[rows, owners, 2] = height - size_z / 2

    def _push_reset_geometry(self):
        if self.cfg['env']['pushTask'].get('reset_geometry_filter') is None:
            return None
        if not hasattr(self,'_push_geometry'):
            from utils.push_reset_geometry import PushResetGeometry
            names=self.gym.get_actor_rigid_body_names(self.envs[0],self.gym.get_actor_handle(self.envs[0],0))
            self._push_geometry=PushResetGeometry(self.cfg['env'],names,self.device)
        return self._push_geometry

    def _apply_push_rsi(self, rows, owners, sample, states, reset_ids, goals):
        from utils.open_push_rsi import lower_push_arms, align_push_pose
        spec = self.cfg['env']['pushTask']; lib = self._motion_lib['push']
        clips = sample['motion_id'].long(); times = sample['time']
        root, rot, dof, vel, ang, dof_vel, _ = lib.get_motion_state(clips, times)
        body, rotations, velocities, angular = lib.get_motion_state_max(clips, times)
        body, rotations, velocities, angular, dof, dof_vel = self._retarget_push_arms(
            body, rotations, velocities, angular, dof, dof_vel, lib)
        local_rows = torch.searchsorted(reset_ids, rows)
        initial = states[local_rows, owners, :3]
        self._box_states[rows, owners, :3] = initial+sample['box_progress'][:, None]*(goals[local_rows, owners]-initial)
        boxes = self._box_states[rows, owners]
        target = boxes[:, :3]+rotate(boxes[:, 3:7], self._push_local[rows, owners])
        forward = rotate(boxes[:, 3:7], boxes.new_tensor([0., 1., 0.]).expand(len(rows), 3))
        tangent = torch.stack((forward[:, 1], -forward[:, 0], torch.zeros(len(rows), device=self.device)), -1)
        targets = target[:, None]+tangent[:, None]*boxes.new_tensor([.2, -.2])[None, :, None]
        body, rotations, velocities, angular, _ = align_push_pose(body, rotations, velocities, angular, targets,
            self._key_body_ids[:2].cpu().tolist(), spec['stand_off_m'])
        if spec.get('rsi_sampling') == 'native_size_cache':
            # Rigid placement only: arbitrary original AMP poses may have a torso
            # ahead of their hands. Move the entire human backward to clear the
            # box rather than rejecting the time or changing any joint/height.
            inv = boxes[:, 3:7].clone(); inv[:, :3] *= -1
            local = rotate(inv[:, None].expand_as(rotations), body-boxes[:, None, :3])
            half = self._box_size[rows, owners]/2
            overlap = (local.abs() < half[:, None]+.08).all(-1)
            shift = torch.where(overlap, local[..., 1]+half[:, None, 1]+.08, 0.).amax(-1).clamp_min(0)
            body = body-forward[:, None]*shift[:, None, None]
            geometry=self._push_reset_geometry()
            if geometry is not None:
                body=body.clone()
                body[...,2]+=geometry.floor_shift(body,rotations)[:,None]
                shift=geometry.rear_shift(body,rotations,boxes,self._box_size[rows,owners])
                body=body-forward[:,None]*shift[:,None,None]
                if geometry.invalid(body,rotations,boxes,self._box_size[rows,owners]).any():
                    raise RuntimeError('PUSH native RSI collider clearance failed')
        limit=spec.get('rsi_max_hand_midpoint_height_m')
        if limit is not None:
            placed_height=body[:,self._key_body_ids[:2],2].mean(-1)
            bad=placed_height>limit+1e-6
            if bad.any():
                examples=torch.stack((clips[bad].float(),times[bad],placed_height[bad]),-1)[:5].cpu().tolist()
                raise RuntimeError('Selected RSI exceeds hand midpoint height limit; do not retarget it: '+str(examples))
        if spec.get('height_mode') == 'amp_reference':
            self._set_amp_midpoint_height(rows, owners, body)
        self._set_pose(rows, owners, body, rotations, velocities, angular, dof, dof_vel)
        loco_rows, loco_owners = self._reset_ref_slots['loco']
        keep = ~((loco_rows[:, None] == rows)&(loco_owners[:, None] == owners)).any(-1)
        self._reset_ref_slots['loco'] = loco_rows[keep], loco_owners[keep]
        self._reset_ref_motion_ids['loco'] = self._reset_ref_motion_ids['loco'][keep]
        self._reset_ref_motion_times['loco'] = self._reset_ref_motion_times['loco'][keep]
        self._reset_ref_slots['push'] = rows, owners
        self._reset_ref_motion_ids['push'] = clips; self._reset_ref_motion_times['push'] = times

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        if skill_name != 'push': return super()._init_amp_obs_ref(slots, motion_ids, motion_times, skill_name)
        from utils.open_push_rsi import lower_push_arms
        from utils.unified_training import append_family
        rows, owners = slots; steps = self._num_amp_obs_steps-1; lib = self._motion_lib['push']
        clips = motion_ids[:, None].expand(-1, steps).reshape(-1)
        times = (motion_times[:, None]-self.dt*torch.arange(1, steps+1, device=self.device)).reshape(-1)
        root, rot, dof, vel, ang, dof_vel, _ = lib.get_motion_state(clips, times)
        body, rotations, velocities, angular = lib.get_motion_state_max(clips, times)
        body, rotations, velocities, angular, dof, dof_vel = self._retarget_push_arms(
            body, rotations, velocities, angular, dof, dof_vel, lib)
        obs = build_amp_observations(root, rot, vel, ang, dof, dof_vel, body[:, self._key_body_ids],
                                    self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets)
        labels = torch.zeros(len(obs), dtype=torch.long, device=self.device)
        self._hist_amp_obs_buf[rows, owners] = append_family(obs, labels).reshape(len(rows), steps, -1)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions); self._push_hits.zero_()

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        self._push_measurements['at'] = self.relation_runtime.done.float().mean().detach()
        for key, value in self._push_measurements.items():
            self._push_metric_sums[key] = self._push_metric_sums.get(key, 0)+value
        self._push_metric_count += 1

    def _physics_step(self):
        spec = self.cfg['env']['pushTask']
        if spec.get('contact_objective') == 'common_face_xy_v1':
            for _ in range(self.control_freq_inv):
                self.render(); self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True); self._refresh_sim_tensors()
                self._push_hits |= common_face_contact(
                    self._rigid_body_pos[:, :, self._key_body_ids[:2]],
                    self._contact_forces[:, :, self._key_body_ids[:2]],
                    self._box_states[:, :2], self._ontop_box_contact[:, :2],
                    self._box_size[:, :2], self._tar_pos, spec)
            return
        if spec.get('contact_objective') == 'midpoint_only_v1':
            for _ in range(self.control_freq_inv):
                self.render(); self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True); self._refresh_sim_tensors()
                if spec.get('require_contact_force', False):
                    self._push_hits |= midpoint_force_contact(
                        self._rigid_body_pos[:, :, self._key_body_ids[:2]],
                        self._contact_forces[:, :, self._key_body_ids[:2]],
                        self._box_states[:, :2], self._ontop_box_contact[:, :2],
                        self._box_size[:, :2], self._push_side, self._push_local,
                        self.cfg['env']['pushPointReward'], spec)
            return
        contact = dict(hand_distance=spec['hand_distance_m'],
                       require_contact_force=spec.get('require_contact_force', True))
        if contact['require_contact_force']:
            contact.update(min_hand_force=spec['min_hand_force_n'], min_box_force=spec['min_box_force_n'])
        for _ in range(self.control_freq_inv):
            self.render(); self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True); self._refresh_sim_tensors()
            self._push_hits |= point_reward_contact(self._rigid_body_pos[:, :, self._key_body_ids[:2]],
                self._contact_forces[:, :, self._key_body_ids[:2]], self._box_states[:, :2], self._ontop_box_contact[:, :2],
                self._box_size[:, :2], contact, self._push_side, self._push_local, self.cfg['env']['pushPointReward'])

    def consume_relation_diagnostics(self):
        result = super().consume_relation_diagnostics()
        result.update({'push/'+key: value/max(self._push_metric_count, 1) for key, value in self._push_metric_sums.items()})
        self._push_metric_sums = {}; self._push_metric_count = 0
        return result
