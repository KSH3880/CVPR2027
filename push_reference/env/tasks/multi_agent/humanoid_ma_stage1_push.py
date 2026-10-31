from pathlib import Path

import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate, quat_mul, quat_from_angle_axis

from push_reference.env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, build_amp_observations
from push_reference.env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
from push_reference.env.tasks.multi_agent.push_reward import PushRuntime, contact_geometry
from push_reference.env.tasks.multi_agent.push_rsi import reference_state
from push_reference.utils.stage1_push import PUSH, push_rows, load_bank
from push_reference.utils.edge_scenario_spec import compose_canonical_graph
from push_reference.utils.edge_ontop_spec import copy_graph_rows
from push_reference.utils.unified_training import append_family, EXPERT_PROBS


class OldTaskRsiCache(SizeRsiCache):
    def __init__(self, env):
        original = env._box_size
        env._box_size = env._rsi_asset_sizes
        try:
            super().__init__(env)
        finally:
            env._box_size = original

    def _build_missing(self):
        self.eligible[self.env._push_env] = 0
        super()._build_missing()


class HumanoidMAStage1Push(HumanoidMACarry):
    _solo_cache_class = OldTaskRsiCache

    def __init__(self, cfg, *args, **kwargs):
        self._push_options = cfg['env']['stage1Push']
        self._push_curriculum_epoch = 0
        navigation = self._push_options.get('navigation')
        spawn_mode = getattr(cfg['args'], 'push_spawn_mode', 'random')
        if spawn_mode not in ('random', 'straight'):
            raise ValueError('PUSH spawn mode must be random or straight')
        if spawn_mode == 'straight' and (not cfg['args'].eval or not navigation):
            raise ValueError('Straight PUSH spawns require PUSH 360 evaluation')
        self._push_straight_spawn = spawn_mode == 'straight'
        configured_mode = self._push_options['rsi'].get('mode', 'corrected')
        self._push_rsi_mode = getattr(cfg['args'], 'push_rsi_mode', '') or configured_mode
        if self._push_rsi_mode not in ('corrected', 'raw'):
            raise ValueError('PUSH RSI mode must be corrected or raw')
        if not cfg['args'].eval and self._push_rsi_mode != configured_mode:
            raise ValueError('PUSH RSI override requires evaluation')
        self._push_horizon = cfg['env']['episodeLength']
        if navigation:
            self._push_horizon = getattr(cfg['args'], 'episode_length', 0) or navigation['episode_length']
        self._amp_num_families = 4
        if cfg['env']['numAgents'] != 2 or cfg['env']['numObjects'] != 4:
            raise ValueError('Stage1 PUSH uses two humans and four object tokens')
        if cfg['env']['numAMPObsSteps'] != 10:
            raise ValueError('PUSH transfer requires ten AMP observation steps')
        if self._push_options['goal_mode'] != 'face_normal':
            raise ValueError('Supported PUSH goal_mode: face_normal')
        super().__init__(cfg, *args, **kwargs)
        self.max_episode_length = max(self.max_episode_length, self._push_horizon)
        if not self._is_eval or 'push' in self._motion_lib:
            if not hasattr(self, '_push_bank'):
                self._push_bank = load_bank(self._push_options, cfg['env']['motion_file'])
            candidates = self._push_bank['candidates']['push']
            remap = getattr(self, '_push_source_ids', None)
            self._push_candidates = torch.tensor([
                [remap.index(v['motion_id']) if remap is not None else v['motion_id'], v['time'], v['box_progress']]
                for v in candidates], device=self.device)
        self._push_metric_sums = torch.zeros(5, device=self.device)
        self._push_metric_count = 0
        self._retention_sums = torch.zeros(4, 2, device=self.device)

    def _motion_skills_to_load(self, available):
        if not self.cfg['args'].eval:
            return available
        from push_reference.utils.evaluation_motion import push_eval_skills
        return push_eval_skills(available, self.cfg['env']['eval'],
                               bool(self._push_env.any()), bool((~self._push_env).any()))

    def _create_motion_lib(self, motion_file, skill):
        if self.cfg['args'].eval and skill == 'push':
            import json
            from pathlib import Path
            from push_reference.env.tasks.multi_agent.push_rsi import EvalPushMotionLib
            bank = json.loads(Path(self._push_options['rsi']['bank']).read_text())
            self._push_source_ids = sorted({row['motion_id'] for row in bank['candidates']['push']})
            self._push_bank = load_bank(self._push_options, motion_file, self._push_source_ids)
            return EvalPushMotionLib(self._push_source_ids, motion_file, skill,
                                     self._dof_body_ids, self._dof_offsets,
                                     self._key_body_ids.cpu().numpy(), self.device)
        return super()._create_motion_lib(motion_file, skill)

    def _load_box_asset(self):
        from push_reference.utils.size_rsi import sample_sizes
        n, m, o = self.num_envs, self.num_agents, self.num_objects
        self._push_env = push_rows(n, self._push_options, self.device, self._task_graph_preset)
        self._box_size = sample_sizes(n, self.device, ('random_scenario' if self._task_graph_preset == 'push' else self._task_graph_preset), self.cfg['env']['box']['build']['taskSizeRanges'], num_agents=m, num_objects=o)
        self._rsi_asset_sizes = self._box_size.clone()
        from push_reference.utils.push_navigation import sample_box_properties
        sizes, masses = sample_box_properties(int(self._push_env.sum())*m, self._push_options['box'], self.device)
        self._box_size[self._push_env, :m] = sizes.reshape(-1, m, 3)
        self._push_box_mass = torch.zeros(n, m, device=self.device)
        self._push_box_mass[self._push_env] = masses.reshape(-1, m)
        self._box_scale = self._box_size/torch.tensor(self._build_base_size, device=self.device)
        density = torch.full((n, o), 100., device=self.device)
        density[self._push_env, :m] = self._push_box_mass[self._push_env]/self._box_size[self._push_env, :m].prod(-1)
        self._box_density = density.flatten()
        self._box_assets = []
        plate = self._push_options['box'].get('bottom_plate')
        push_slots = (self._push_env[:, None] & (torch.arange(o, device=self.device) < m)).flatten().cpu().tolist()
        for index, (size, rho) in enumerate(zip(self._box_size.reshape(-1, 3).cpu().tolist(), self._box_density.cpu().tolist())):
            options = gymapi.AssetOptions()
            options.angular_damping = options.linear_damping = .01
            options.max_angular_velocity = 100.
            options.density = rho
            options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            if plate and push_slots[index]:
                from push_reference.utils.push_box_asset import create_plate_box
                mass = rho*size[0]*size[1]*size[2]
                directory = Path(self.cfg['args'].output_path)/'push_assets'
                asset = create_plate_box(self.gym, self.sim, size, mass, options, plate, directory, index)
            else:
                asset = self.gym.create_box(self.sim, *size, options)
            self._box_assets.append(asset)

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        if self._push_env[env_id] and object_id < self.num_agents:
            handle = self._box_handles[-1]
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for prop in props:
                prop.friction = self._push_options['box']['friction']
                prop.restitution = 0.
            plate = self._push_options['box'].get('bottom_plate')
            if plate:
                props[1].friction = plate['friction']
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self._push_hits = torch.zeros(self.num_envs, self.num_agents, device=self.device, dtype=torch.bool)
        self._push_interaction = torch.zeros_like(self._push_hits)
        self._push_motion_ids = torch.zeros_like(self._push_hits, dtype=torch.long)
        self._push_motion_times = torch.zeros_like(self._push_hits, dtype=torch.float)
        self._push_rsi_retreat = torch.zeros_like(self._push_motion_times)
        if self._push_options['rsi'].get('clearance') is not None:
            from push_reference.env.tasks.multi_agent.push_rsi import collision_shapes
            names = self._motion_lib['push']._motions[0].skeleton_tree.node_names if 'push' in self._motion_lib else None
            if names is not None:
                self._push_collision_shapes = collision_shapes('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml', names, self.device)
        self.relation_runtime = PushRuntime(self, self.relation_runtime)

    def _reset_ontop_context_envs(self, ids):
        if not len(ids):
            return
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        old, push = ids[~self._push_env[ids]], ids[self._push_env[ids]]
        if len(old):
            super()._reset_ontop_context_envs(old)
        if len(push):
            self._reset_push(push)

    def _reset_push(self, ids):
        k, m = len(ids), self.num_agents
        graph = compose_canonical_graph(torch.full((k, m), 3, device=self.device, dtype=torch.long))
        graph.edge_relation[graph.edge_relation == 6] = PUSH
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._sample_box_assignments(ids)
        self._push_hits[ids] = False
        self._push_rsi_retreat[ids] = 0
        self._push_interaction[ids] = torch.rand(k, m, device=self.device) < self._push_options['rsi']['interaction_probability']
        if self._is_eval:
            skills = self.cfg['env']['eval']['skill']
            probabilities = torch.tensor(self.cfg['env']['eval']['skillInitProb'], device=self.device)
            if any(skill not in ('loco', 'push') for skill in skills) and self._push_env[ids].any():
                raise ValueError('PUSH evaluation starts support loco and push')
            push_probability = sum(float(weight) for skill, weight in zip(skills, probabilities) if skill == 'push')/float(probabilities.sum())
            self._push_interaction[ids] = torch.rand(k, m, device=self.device) < push_probability
        yaw = torch.rand(k, m, device=self.device)*2*torch.pi
        q = quat_from_angle_axis(yaw.flatten(), torch.tensor([0., 0., 1.], device=self.device).expand(k*m, -1)).reshape(k, m, 4)
        normal = quat_rotate(q.reshape(-1, 4), torch.tensor([0., 1., 0.], device=self.device).expand(k*m, -1)).reshape(k, m, 3)
        source = self._env_origins[ids, None].expand(-1, m, -1).clone()
        source[..., 0] += torch.tensor([-6., 6.], device=self.device)
        source[..., 2] = self._box_size[ids, :m, 2]/2
        lo, hi = self._push_options['goal_distance_range']
        navigation = self._push_options.get('navigation')
        if navigation:
            from push_reference.utils.push_navigation import curriculum
            _, stage = curriculum(self._push_options, self._push_curriculum_epoch, self._is_eval)
            hi = stage['goal_max']
        rsi_hi = hi
        if self._push_options['rsi'].get('uniform_goal_distance', False):
            rsi_hi = self._push_options['rsi'].get('goal_distance_max', hi)
        if navigation:
            box_options = self._push_options['box']
            if 'size_ranges' in box_options:
                max_edge = max(axis[1] for axis in box_options['size_ranges'][:2])
            else:
                max_edge = box_options['size_range'][1] if 'size_range' in box_options else max(box_options['size'][:2])
            half_span = max(hi, rsi_hi)+max_edge
            source[..., 0] = self._env_origins[ids, None, 0]+half_span*torch.tensor([-1.,1.], device=self.device)
        goal_max = hi if rsi_hi == hi else torch.where(self._push_interaction[ids], rsi_hi, hi)
        distance = lo+(goal_max-lo)*torch.rand(k, m, device=self.device)
        self._tar_pos[ids] = source+distance[..., None]*normal
        self._box_states[ids, :m, :3] = source
        self._box_states[ids, :m, 3:7] = q
        self._box_states[ids, :m, 7:] = 0
        for spare in range(m, self.num_objects):
            self._box_states[ids, spare] = 0
            self._box_states[ids, spare, :3] = self._env_origins[ids]
            self._box_states[ids, spare, 1] += 9. if spare == m else -9.
            self._box_states[ids, spare, 2] = self._box_size[ids, spare, 2]/2
            self._box_states[ids, spare, 6] = 1
        self._platform_pos[ids] = self._platform_default_pos[ids]
        self._tar_platform_pos[ids] = self._tar_platform_default_pos[ids]
        rows, owners = self._expand_slots(ids)
        interaction = self._push_interaction[rows, owners]
        order = ((True, 'push'), (False, 'push_loco')) if navigation else ((False, 'push_loco'), (True, 'push'))
        for use_push, skill in order:
            chosen = interaction == use_push
            r, a = rows[chosen], owners[chosen]
            if not len(r):
                continue
            lib = self._motion_lib[skill]
            if use_push:
                clips, times, progress = self._sample_push_rsi(len(r))
                p, rot, vel, ang, dof, dof_vel = reference_state(lib, clips, times, self._dof_body_ids, self._dof_offsets,
                    correct_hand_height=self._push_rsi_mode != 'raw')
                self._push_motion_ids[r, a], self._push_motion_times[r, a] = clips, times
                if not self._push_options['rsi'].get('uniform_goal_distance', False):
                    self._box_states[r, a, :3] += progress[:, None]*(self._tar_pos[r, a]-self._box_states[r, a, :3])
            else:
                clips = lib.sample_motions(len(r))
                times = lib.sample_time_rsi(clips)
                _, _, dof, _, _, dof_vel, _ = lib.get_motion_state(clips, times)
                p, rot, vel, ang = lib.get_motion_state_max(clips, times)
            self._push_motion_ids[r, a], self._push_motion_times[r, a] = clips, times
            hands = p[:, self._key_body_ids[:2]]
            forward = hands.mean(1)-p[:, 0]
            direction = quat_rotate(self._box_states[r, a, 3:7], torch.tensor([0., 1., 0.], device=self.device).expand(len(r), -1))
            angle = torch.atan2(direction[:, 1], direction[:, 0])
            angle -= torch.atan2(forward[:, 1], forward[:, 0]) if use_push else torch_utils_heading(rot[:, 0])
            spawn = None
            if navigation and not use_push:
                from push_reference.utils.push_navigation import sample_spawns
                spawn = sample_spawns(self._box_states[ids], self._box_size[ids], normal,
                    self._humanoid_root_states[ids, :, :3], self._push_interaction[ids],
                    (navigation['spawn_radius_range'][0], stage['radius_max']),
                    0. if self._push_straight_spawn else stage['half_angle_degrees'], navigation['clearance'])
                selected = spawn.reshape(-1, 3)[chosen]
                toward_box = self._box_states[r, a, :2]-selected[:, :2]
                angle = torch.atan2(toward_box[:, 1], toward_box[:, 0])-torch_utils_heading(rot[:, 0])
            turn = quat_from_angle_axis(angle, torch.tensor([0., 0., 1.], device=self.device).expand(len(r), -1))
            point = self._box_states[r, a, :3]-(self._box_size[r, a, 1]/2)[:, None]*quat_rotate(self._box_states[r, a, 3:7], torch.tensor([0., 1., 0.], device=self.device).expand(len(r), -1))
            rotated = quat_rotate(turn[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4), p.reshape(-1, 3)).reshape_as(p)
            align = rotated[:, self._key_body_ids[:2]].mean(1) if use_push else rotated[:, 0]
            stand_off = self._push_options['rsi']['stand_off'] if use_push else 1.+torch.rand(len(r), device=self.device)
            direction = quat_rotate(self._box_states[r, a, 3:7], torch.tensor([0., 1., 0.], device=self.device).expand(len(r), -1))
            target = point-direction*(stand_off if isinstance(stand_off, float) else stand_off[:, None])
            if spawn is not None:
                target = selected
            shift = target-align
            shift[:, 2] = 0
            p = rotated+shift[:, None]
            rot = quat_mul(turn[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4), rot.reshape(-1, 4)).reshape_as(rot)
            if use_push and hasattr(self, '_push_collision_shapes'):
                from push_reference.env.tasks.multi_agent.push_rsi import face_retreat
                retreat = face_retreat(p, rot, point, direction, self._push_collision_shapes,
                                       self._push_options['rsi']['clearance'])
                p -= retreat[:, None, None]*direction[:, None]
                self._push_rsi_retreat[r, a] = retreat
            vel = quat_rotate(turn[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4), vel.reshape(-1, 3)).reshape_as(vel)
            ang = quat_rotate(turn[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4), ang.reshape(-1, 3)).reshape_as(ang)
            self._set_env_state(r, a, p[:, 0], rot[:, 0], dof, vel[:, 0], ang[:, 0], dof_vel)
            self._kinematic_humanoid_rigid_body_states[r, a] = torch.cat((p, rot, vel, ang), -1)
            self._every_env_init_dof_pos[r, a] = dof
        self._reset_env_tensors(ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(ids)
        self._compute_observations(ids)
        self._compute_amp_observations(ids)
        self._hist_amp_obs_buf[ids] = self._curr_amp_obs_buf[ids, :, None]
        self._push_history(ids)

    def _sample_push_rsi(self, count):
        records = self._push_candidates
        clips = records[:, 0].long().unique()
        weights = self._motion_lib['push']._motion_weights[clips]
        chosen = clips[torch.multinomial(weights, count, replacement=True)]
        result = []
        late = torch.rand(count, device=self.device) < self._push_options['rsi']['late_probability']
        for clip, use_late in zip(chosen.tolist(), late.tolist()):
            pool = records[records[:, 0] == clip]
            pool = pool[pool[:, 1].argsort()]
            if use_late:
                pool = pool[-max(1, (len(pool)*3+9)//10):]
            result.append(pool[torch.randint(len(pool), (), device=self.device)])
        result = torch.stack(result)
        return result[:, 0].long(), result[:, 1], result[:, 2]

    def _push_history(self, ids):
        r, a = self._expand_slots(ids)
        for use_push, skill in ((False, 'push_loco'), (True, 'push')):
            mask = self._push_interaction[r, a] == use_push
            rows, owners = r[mask], a[mask]
            if not len(rows):
                continue
            if not use_push or self._push_rsi_mode == 'raw':
                self._init_amp_obs_ref((rows, owners), self._push_motion_ids[rows, owners], self._push_motion_times[rows, owners], skill)
                continue
            clips = self._push_motion_ids[rows, owners].repeat_interleave(9)
            times = (self._push_motion_times[rows, owners, None]-self.dt*torch.arange(1, 10, device=self.device)).flatten()
            p, rot, vel, ang, dof, dof_vel = reference_state(self._motion_lib[skill], clips, times, self._dof_body_ids, self._dof_offsets)
            obs = build_amp_observations(p[:, 0], rot[:, 0], vel[:, 0], ang[:, 0], dof, dof_vel, p[:, self._key_body_ids], self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets)
            self._hist_amp_obs_buf[rows, owners] = append_family(obs, torch.full_like(clips, 3), 4).reshape(len(rows), 9, 133)

    def _push_state(self, ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        box = self._box_states[ids, :self.num_agents]
        hands = self._rigid_body_pos[ids][:, :, self._key_body_ids[:2]]
        forces = self._contact_forces[ids][:, :, self._key_body_ids[:2]]
        phi, hit, point, normal = contact_geometry(hands, box, self._box_size[ids, :self.num_agents], forces,
            self._ontop_box_contact[ids, :self.num_agents], midpoint_only=self._push_options.get('contact_midpoint_only', False),
            contact_height=self._push_options.get('contact_height', .6175),
            face_xy=self._push_options.get('contact_mode') == 'face_xy')
        diff = box[..., :3]-self._tar_pos[ids]
        roots = self._humanoid_root_states[ids, :, :3]
        return dict(phi_contact=phi, hit=hit, point=point, normal=normal, phi_at=torch.exp(-10*diff.square().sum(-1)),
            distance_xy=diff[..., :2].norm(dim=-1),
            z_error=diff[..., 2], distance=diff.norm(dim=-1),
            progress_contact=1/(1+(roots[..., :2]-box[..., :2]).norm(dim=-1).sub(.5).clamp_min(0)),
            progress_at=1/(1+diff[..., :2].norm(dim=-1).sub(.1).clamp_min(0)))

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        graph = self.relation_runtime.graph
        owner = graph.edge_owner[ids]
        primary = graph.edge_relation[ids] == PUSH
        mask = self._push_env[ids, None] & graph.edge_valid[ids]
        data = self._push_state(ids)
        navigation = self._push_options.get('navigation')
        if navigation:
            from push_reference.utils.push_navigation import approach_distance
            distance = approach_distance(self._humanoid_root_states[ids, :, :3],
                self._box_states[ids, :self.num_agents, :3], data['normal'],
                self._box_size[ids, :self.num_agents]/2, navigation['approach_stand_off'],
                navigation['clearance'])
            data['progress_contact'] = 1/(1+distance)
        phi = torch.where(mask, torch.where(primary, data['phi_contact'].gather(1, owner), data['phi_at'].gather(1, owner)), phi)
        diag['progress'] = torch.where(mask, torch.where(primary, data['progress_contact'].gather(1, owner), data['progress_at'].gather(1, owner)), diag['progress'])
        diag['z_error'] = torch.where(mask, data['z_error'].gather(1, owner), diag['z_error'])
        diag['distance'] = torch.where(mask, data['distance'].gather(1, owner), diag['distance'])
        return phi, diag

    def _compute_reset(self):
        super()._compute_reset()
        if self._push_options.get('navigation'):
            horizon = torch.where(self._push_env, self._push_horizon, self.cfg['env']['episodeLength'])
            fallen = self._terminate_buf.reshape(self.num_envs, self.num_agents).bool().any(-1)
            self.reset_buf.copy_((self.progress_buf >= horizon-1) | fallen)

    def _compute_agent_collision_violation(self, root_pos, min_dist):
        result = super()._compute_agent_collision_violation(root_pos, min_dist)
        distance = (root_pos[:, 0, :2]-root_pos[:, 1, :2]).norm(dim=-1)
        return torch.where(self._push_env[:, None], ((min_dist-distance).clamp_min(0)/min_dist)[:, None], result)

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        terms = self.extras['reward_terms'].view(self.num_envs, self.num_agents, 7)
        correction = torch.zeros_like(terms)
        correction[self._push_env, :, 6] = -terms[self._push_env, :, 5]
        correction[self._push_env, :, 5] = -terms[self._push_env, :, 5]
        terms += correction
        self.rew_buf.copy_(terms[..., 6].flatten())
        self._reward_term_sums += correction.sum((0, 1))
        if hasattr(self, '_retention_sums'):
            packet = self.relation_runtime.suffix().reshape(self.num_envs, self.num_agents, 5)
            for task_id in range(4):
                selected = (packet[..., 1] == task_id) & (~self._push_env[:, None])
                self._retention_sums[task_id, 0] += (self.relation_runtime.done & selected).sum()
                self._retention_sums[task_id, 1] += selected.sum()
        if hasattr(self, '_push_metric_sums') and self._push_env.any():
            at = self.relation_runtime.done[self._push_env]
            interaction = self._push_interaction[self._push_env]
            self._push_metric_sums += torch.stack((at.float().sum(), self._push_hits[self._push_env].float().sum(), interaction.float().sum(), (at & interaction).float().sum(), (at & ~interaction).float().sum()))
            self._push_metric_count += at.numel()

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._push_hits.zero_()

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            self._push_hits |= self._push_state()['hit'] & self._push_env[:, None]

    def _amp_family(self, env_ids=None, slot_agent=None):
        family = super()._amp_family(env_ids, slot_agent)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        rows = ids.repeat_interleave(self.num_agents) if slot_agent is None else ids
        return torch.where(self._push_env[rows], 3, family)

    def _fetch_conditioned_amp_demo(self, count):
        old_fraction = float((~self._push_env).float().mean())
        family = torch.multinomial(torch.tensor([.65*old_fraction, .1*old_fraction, .25*old_fraction, 1-old_fraction], device=self.device), count, replacement=True)
        weights = torch.zeros(4, 10, device=self.device)
        weights[:3, :8] = torch.tensor(EXPERT_PROBS, device=self.device)
        weights[3, 9], weights[3, 8] = .33, .67
        skills = torch.multinomial(weights[family], 1).flatten()
        result = torch.empty(count, self.get_num_amp_obs(), device=self.device)
        for index, name in enumerate(list(self.cfg['env']['skill'])+['push', 'push_loco']):
            selected = (skills == index).nonzero().flatten()
            if not len(selected):
                continue
            lib = self._motion_lib[name]
            clips = lib.sample_motions(len(selected))
            truncate = self.dt*9
            times = lib.sample_time(clips, truncate_time=truncate)+truncate
            obs = self.build_amp_obs_demo(clips, times, lib).reshape(len(selected), 10, -1)
            result[selected] = append_family(obs, family[selected, None].expand(-1, 10), 4).flatten(1)
        return result

    def consume_relation_diagnostics(self):
        result = super().consume_relation_diagnostics()
        count = max(self._push_metric_count, 1)
        s = self._push_metric_sums
        result.update({'push/current_goal_success': s[0]/count, 'push/contact': s[1]/count,
            'push/rsi_current_success': s[3]/s[2].clamp_min(1),
            'push/loco_current_success': s[4]/(count-s[2]).clamp_min(1),
            'push/transition_fraction': self._push_env.float().mean()})
        for index, name in enumerate(('sit', 'climb', 'carry_at', 'carry_ontop')):
            result['retention/'+name+'/current_success'] = self._retention_sums[index, 0]/self._retention_sums[index, 1].clamp_min(1)
        if self._push_options.get('navigation'):
            from push_reference.utils.push_navigation import curriculum
            index, stage = curriculum(self._push_options, self._push_curriculum_epoch, self._is_eval)
            values = {'push/curriculum_stage': index, 'push/goal_max_m': stage['goal_max'],
                'push/spawn_radius_max_m': stage['radius_max'],
                'push/spawn_half_angle_degrees': 0. if self._push_straight_spawn else stage['half_angle_degrees']}
            result.update({key: torch.as_tensor(value, device=self.device) for key, value in values.items()})
        self._retention_sums.zero_()
        self._push_metric_sums.zero_()
        self._push_metric_count = 0
        return result

    def evaluation_context(self):
        packet = self.relation_runtime.suffix().reshape(self.num_envs, self.num_agents, 5)
        return packet[..., 1].long().clone(), self._push_interaction.clone()

    def evaluation_metrics(self, final, ever, context):
        tasks, interaction = context
        result = {}
        names = ('sit', 'climb', 'carry_at', 'carry_ontop', 'push')
        for index, name in enumerate(names):
            mask = tasks == index
            count = int(mask.sum())
            result[name] = dict(agent_trials=count,
                final_success=float(final[mask].float().mean()) if count else None,
                ever_success=float(ever[mask].float().mean()) if count else None)
        for name, start in (('push_loco', ~interaction), ('push_rsi', interaction)):
            mask = (tasks == 4) & start
            count = int(mask.sum())
            result[name] = dict(agent_trials=count,
                final_success=float(final[mask].float().mean()) if count else None,
                ever_success=float(ever[mask].float().mean()) if count else None)
        return {'task_goal_success': result}


def torch_utils_heading(rotation):
    from push_reference.utils.torch_utils import calc_heading
    return calc_heading(rotation)
