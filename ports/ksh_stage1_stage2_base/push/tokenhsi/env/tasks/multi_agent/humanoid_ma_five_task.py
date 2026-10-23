"""Existing four Stage1 tasks plus PUSH_AT in the same policy and sampler."""
import math
from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from utils.edge_context_spec import PUSH_CONTACT, AT
from utils.edge_ontop_spec import select_graph
from utils.edge_scenario_spec import classify_templates
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.push_point import common_face_xy, common_face_contact
from utils.push_motion_metrics import PushMotionMetrics
from utils.stage1_push import sample_boxes_and_goals, sample_box_sizes
from utils.five_task import demo_distribution


class FiveTaskRuntime(Stage1ContextRuntime):
    def __init__(self, task, *args):
        super().__init__(*args)
        self.task, self.rows = task, None

    def reset(self, ids, *args, **kwargs):
        self.rows = ids
        try:
            return super().reset(ids, *args, **kwargs)
        finally:
            self.rows = None

    def success(self, phi, z_error, feet_error, graph, region_error):
        original = super().success(phi, z_error, feet_error, graph, region_error)
        hits = self.task._push_hits if self.rows is None else self.task._push_hits[self.rows]
        return torch.where(graph.edge_relation == PUSH_CONTACT,
                           hits.gather(1, graph.edge_owner), original) & graph.edge_valid


class HumanoidMAFiveTask(HumanoidMAStage1Push):
    # Native cache for the original four tasks; PUSH has its own screened cache.
    from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache as _solo_cache_class
    _reset_envs = HumanoidMACarry._reset_envs
    _init_amp_obs_ref = HumanoidMACarry._init_amp_obs_ref

    def _push_active_slots(self):
        graph = self.relation_runtime.graph
        from env.tasks.multi_agent.edge_context_reward import owner_sum
        return owner_sum((graph.edge_valid & (graph.edge_relation == PUSH_CONTACT)).float(), graph) > 0

    def _physics_step(self):
        # Match the existing common-face physics path exactly; observe each tick.
        if self.cfg['env']['pushTask'].get('contact_objective') != 'common_face_xy_v1':
            return super()._physics_step()
        if not hasattr(self, '_push_motion_metrics'):
            self._push_motion_metrics = PushMotionMetrics((self.num_envs, self.num_agents), self.device)
        metrics = self._push_motion_metrics
        metrics.begin_step()
        active = self._push_active_slots()
        for _ in range(self.control_freq_inv):
            before = self._box_states[:, :self.num_agents, :3].clone()
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            contact = common_face_contact(
                self._rigid_body_pos[:, :, self._key_body_ids[:2]],
                self._contact_forces[:, :, self._key_body_ids[:2]],
                self._box_states[:, :self.num_agents], self._ontop_box_contact[:, :self.num_agents],
                self._box_size[:, :self.num_agents], self._tar_pos, self.cfg['env']['pushTask'])
            self._push_hits |= contact
            metrics.tick(before, self._box_states[:, :self.num_agents, :3],
                         self._tar_pos, contact, active)

    def _compute_reward(self, actions):
        HumanoidMACarry._compute_reward(self, actions)
        if hasattr(self, '_push_motion_metrics'):
            graph = self.relation_runtime.graph
            from env.tasks.multi_agent.edge_context_reward import owner_sum
            at = owner_sum((self.relation_runtime.own_success & graph.edge_valid &
                            (graph.edge_relation == AT)).float(), graph) > 0
            distance = (self._box_states[:, :self.num_agents, :2] - self._tar_pos[..., :2]).norm(dim=-1)
            self._push_motion_metrics.finish_step(self._push_active_slots(), self._push_hits,
                at, distance, self.dt)

    def _finish_relation_diagnostics(self):
        HumanoidMACarry._finish_relation_diagnostics(self)
        if hasattr(self, '_push_motion_metrics'):
            self._push_motion_metrics.finish_episodes(self.reset_buf.bool(), self._push_active_slots())

    def consume_relation_diagnostics(self):
        result = HumanoidMACarry.consume_relation_diagnostics(self)
        if hasattr(self, '_push_motion_metrics'):
            result.update(self._push_motion_metrics.consume())
        return result

    def _load_box_asset(self):
        from utils.size_rsi import sample_sizes
        env = self.cfg['env']
        n, m, o = self.num_envs, self.num_agents, self.num_objects
        preset = self._task_graph_preset
        native_preset = 'holding_at' if preset == 'push_at' else preset
        sizes = sample_sizes(n, self.device, native_preset, env['box']['build']['taskSizeRanges'],
                             num_agents=m, num_objects=o)
        p = env['relationGraph']['template_probabilities']['PUSH_AT']
        mask = torch.rand(n, m, device=self.device) < p
        if preset != 'random_scenario':
            mask[:] = preset == 'push_at'
        candidates = sample_box_sizes(n, self.device, env)
        from utils.push_eval_sizes import override_sizes, save_report
        candidates = override_sizes(self, candidates)
        if hasattr(self, '_push_eval_size_report'):
            self._push_eval_asset_mask = mask
        sizes[:, :m] = torch.where(mask[..., None], candidates[:, :m], sizes[:, :m])
        self._box_size = sizes
        self._box_scale = sizes / torch.tensor(self._build_base_size, device=self.device)
        density = torch.full((n, o), 100., device=self.device)
        density[:, :m] = torch.where(mask, env['pushBox']['mass']/sizes[:, :m].prod(-1), density[:, :m])
        self._box_density = density.flatten()
        self._box_assets = []
        assets = {}
        for index, dimensions in enumerate(sizes.reshape(-1, 3).cpu().tolist()):
            is_push = index % o < m and bool(mask[index//o, index % o])
            key = tuple(dimensions), is_push
            if key not in assets:
                options = gymapi.AssetOptions()
                options.density = env['pushBox']['mass']/math.prod(dimensions) if is_push else 100.
                options.angular_damping = options.linear_damping = .01
                options.max_angular_velocity = 100.
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                asset = self.gym.create_box(self.sim, *dimensions, options)
                if is_push:
                    props = self.gym.get_asset_rigid_shape_properties(asset)
                    for prop in props:
                        prop.friction = env['pushBox']['friction']
                        prop.restitution = env['pushBox']['restitution']
                    self.gym.set_asset_rigid_shape_properties(asset, props)
                assets[key] = asset
            self._box_assets.append(assets[key])
        save_report(self)

    def _init_ontop_context_runtime(self):
        # Initialize original buffers, then replace only the mixed-task runtime.
        HumanoidMAStage1Push._init_ontop_context_runtime(self)
        self.relation_runtime = FiveTaskRuntime(self, self.num_envs, self.relation_runtime.graph,
                                                self._relation_cfg, self.device)

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = HumanoidMACarry._evaluate_ontop_context(self, env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        bodies = self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        boxes, sizes = self._box_states[ids, :self.num_agents], self._box_size[ids, :self.num_agents]
        hand_distances, _, _ = common_face_xy(bodies[:, :, self._key_body_ids[:2]], boxes, sizes, self._tar_pos[ids])
        hand_distance = hand_distances.amax(-1)
        score = torch.exp(-10 * hand_distance.square())
        if env_ids is not None:
            self._push_hits[ids] = False
        graph = select_graph(self.relation_runtime.graph, ids)
        mask = graph.edge_valid & (graph.edge_relation == PUSH_CONTACT)
        phi = torch.where(mask, score.gather(1, graph.edge_owner), phi)
        for key in ('distance', 'distance_xy'):
            diag[key] = torch.where(mask, hand_distance.gather(1, graph.edge_owner), diag[key])
        diag['z_error'] = torch.where(mask, torch.zeros_like(diag['z_error']), diag['z_error'])
        distance = (bodies[:, :, 0, :2] - boxes[..., :2]).norm(dim=-1)
        progress = 1/(1+(distance-.5).clamp_min(0))
        diag['progress'] = torch.where(mask, progress.gather(1, graph.edge_owner), diag['progress'])
        return phi, diag

    def _reset_ref_state_init(self, rows, owners):
        templates = classify_templates(self.relation_runtime.graph, rows, owners)
        push = templates == 5
        if (~push).any():
            HumanoidMACarry._reset_ref_state_init(self, rows[~push], owners[~push])
        if push.any():
            r, a = rows[push], owners[push]
            lib = self._motion_lib['loco']
            clips = lib.sample_motions(len(r))
            times = lib.sample_time_rsi(clips)
            _, _, dof, _, _, dof_vel, _ = lib.get_motion_state(clips, times)
            body, rot, vel, ang = lib.get_motion_state_max(clips, times)
            body = body + (self._env_origins[r] + self._agent_spawn_offsets[a])[:, None]
            self._set_pose(r, a, body, rot, vel, ang, dof, dof_vel)
            for store, values in ((self._reset_ref_slots, (r, a)),
                                  (self._reset_ref_motion_ids, clips), (self._reset_ref_motion_times, times)):
                if 'loco' not in store:
                    store['loco'] = values
                elif isinstance(values, tuple):
                    store['loco'] = tuple(torch.cat((old, new)) for old, new in zip(store['loco'], values))
                else:
                    store['loco'] = torch.cat((store['loco'], values))

    def _reset_task(self, ids):
        HumanoidMACarry._reset_task(self, ids)
        self._reset_push_slots(ids)

    def _reset_push_slots(self, ids):
        if hasattr(self, '_push_motion_metrics'):
            self._push_motion_metrics.reset(ids)
        rows, owners = self._expand_slots(ids)
        chosen = classify_templates(self.relation_runtime.graph, rows, owners) == 5
        rows, owners = rows[chosen], owners[chosen]
        self._push_hits[ids] = False
        self._push_rsi_requested[ids] = False
        self._push_rsi_interaction[ids] = False
        if not len(rows):
            return
        states, goals, forward, distances = sample_boxes_and_goals(len(ids), self.device,
            self._env_origins[ids], self.cfg['env'], self._box_size[ids])
        local = torch.searchsorted(ids, rows)
        self._box_states[rows, owners] = states[local, owners]
        self._tar_pos[rows, owners] = goals[local, owners]
        self._platform_pos[rows, owners] = self._platform_default_pos[rows, owners]
        self._tar_platform_pos[rows, owners] = self._tar_platform_default_pos[rows, owners]
        lib = self._motion_lib['loco']
        # Use the actual reset clip/time in AMP history as well as in the pose.
        lr, la = self._reset_ref_slots['loco']
        match = (lr[:, None] == rows) & (la[:, None] == owners)
        indices = match.long().argmax(0)
        clips, times = self._reset_ref_motion_ids['loco'][indices], self._reset_ref_motion_times['loco'][indices]
        _, root_rot, dof, _, _, dof_vel, _ = lib.get_motion_state(clips, times)
        body, rotations, velocities, angular = lib.get_motion_state_max(clips, times)
        spec = self.cfg['env']['pushTask']
        from utils.push_random_spawn import MODE
        random_spawn = spec.get('loco_spawn_mode') == MODE
        target_yaw = (torch.rand(len(rows), device=self.device)*2*math.pi if random_spawn
                      else yaw_from_quat(states[local, owners, 3:7])+math.pi/2)
        angle = target_yaw-yaw_from_quat(root_rot)
        q = torch.zeros_like(root_rot)
        q[:, 2], q[:, 3] = (angle/2).sin(), (angle/2).cos()
        qs = q[:, None].expand_as(rotations)
        relative = rotate(qs, body-body[:, :1])
        rotations, velocities, angular = multiply(qs, rotations), rotate(qs, velocities), rotate(qs, angular)
        boxes, sizes = self._box_states[rows, owners], self._box_size[rows, owners]
        direction = forward[local, owners]
        geometry = self._push_reset_geometry()
        if random_spawn:
            from utils.push_random_spawn import random_approach
            placed = random_approach(geometry, relative, rotations, boxes, sizes, body[:, 0, 2], spec)
        else:
            placed = self._rear_loco_placement(relative, rotations, boxes, sizes, body[:, 0, 2], direction, geometry)
        self._set_pose(rows, owners, placed, rotations, velocities, angular, dof, dof_vel)
        probability = self.cfg['env']['pushTask']['interaction_probability']
        if self._is_eval:
            evaluation = self.cfg['env']['eval']
            probability = sum(p for skill, p in zip(evaluation['skill'], evaluation['skillInitProb'])
                              if skill == 'push') / sum(evaluation['skillInitProb'])
        interaction = torch.rand(len(rows), device=self.device) < probability
        self._push_rsi_requested[rows, owners] = interaction
        self._push_rsi_interaction[rows, owners] = interaction
        if interaction.any():
            r, a = rows[interaction], owners[interaction]
            sample = self._push_native_rsi.sample(self._box_size[r, a])
            travel = distances[local[interaction], a]
            limit = 1-(self.cfg['env']['pushTask']['minimum_rsi_remaining_m']+.001)/travel
            sample['box_progress'] = torch.minimum(sample['box_progress'], limit)
            for key, value in self._push_rsi_samples.items():
                value[r, a] = sample[key]
            self._apply_push_rsi(r, a, sample, states, ids, goals)
            if self.cfg['env']['pushGoalRandomization'].get('distance_reference') == 'reset_box_center':
                # Keep the requested distance from the actual RSI reset box,
                # rather than shortening it when RSI advances the box.
                index = local[interaction]
                self._tar_pos[r, a] = self._box_states[r, a, :3]+goals[index, a]-states[index, a, :3]

    def _rear_loco_placement(self, relative, rotations, boxes, sizes, root_z, direction, geometry):
        lo, hi = self.cfg['env']['pushTask']['loco_distance_range_m']
        position = boxes[:, :3].clone()
        position[:, 2] = root_z
        for _ in range(64):
            distance = lo+(hi-lo)*torch.rand(len(boxes), device=self.device)
            candidate = position.clone()
            candidate[:, :2] = boxes[:, :2] - direction[:, :2]*distance[:, None]
            world = relative + candidate[:, None]
            if geometry is not None:
                world[..., 2] += geometry.floor_shift(world, rotations)[:, None]
                invalid = geometry.invalid(world, rotations, boxes, sizes)
            else:
                raise ValueError('PUSH mixed task requires collider-safe reset')
            if _ == 0:
                placed = world.clone()
                pending = invalid
            else:
                placed[pending] = world[pending]
                pending &= invalid
            if not pending.any():
                break
        else:
            raise RuntimeError('Cannot place PUSH loco without box penetration')
        return placed

    def _ontop_scene_infeasible(self, ids):
        invalid = HumanoidMACarry._ontop_scene_infeasible(self, ids)
        from utils.push_random_spawn import MODE, unsafe_initial_rows
        spec = self.cfg['env']['pushTask']
        if spec.get('loco_spawn_mode') != MODE:
            return invalid
        rows, owners = self._expand_slots(ids)
        chosen = classify_templates(self.relation_runtime.graph, rows, owners) == 5
        rows, owners = rows[chosen], owners[chosen]
        if len(rows):
            poses = self._kinematic_humanoid_rigid_body_states[rows, owners]
            # Native RSI intentionally starts near CONTACT; retain its screened
            # clearance. Loco starts require additional free space at every box.
            minimum = torch.where(self._push_rsi_interaction[rows, owners],
                                  spec['reset_geometry_margin_m'], spec['loco_other_box_clearance_m'])
            bad = unsafe_initial_rows(self._push_reset_geometry(), poses,
                                      self._box_states[rows], self._box_size[rows], minimum)
            bad_rows = torch.zeros(len(ids), device=self.device, dtype=torch.long)
            bad_rows.scatter_add_(0, torch.searchsorted(ids, rows), bad.long())
            invalid |= bad_rows > 0
        return invalid

    def _fetch_conditioned_amp_demo(self, count):
        from utils.unified_training import append_family
        probs, experts = demo_distribution(self.cfg['env'])
        family = torch.multinomial(torch.tensor(probs, device=self.device), count, replacement=True)
        skill_ids = torch.multinomial(torch.tensor(experts, device=self.device)[family], 1).flatten()
        result = torch.empty(count, self.get_num_amp_obs(), device=self.device)
        for uid, name in enumerate(self.cfg['env']['skill']):
            indices = (skill_ids == uid).nonzero().flatten()
            if not len(indices):
                continue
            lib = self._motion_lib[name]
            clips = lib.sample_motions(len(indices))
            history = self.dt*(self._num_amp_obs_steps-1)
            times = lib.sample_time(clips, truncate_time=history)+history
            obs = self.build_amp_obs_demo(clips, times, lib).reshape(len(indices), self._num_amp_obs_steps, -1)
            labels = family[indices, None].expand(-1, self._num_amp_obs_steps)
            result[indices] = append_family(obs, labels, self._amp_num_families).flatten(1)
        return result
