import torch
from isaacgym import gymapi

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
from env.tasks.multi_agent.jointpush_reward import Stage1PushRuntime, contact_state, gate_push_at_progress
from utils.stage1_jointpush import PUSH, compose_graph, sample_assets, box_densities
from utils.edge_scenario_spec import classify_templates
from utils.edge_ontop_spec import copy_graph_rows
from utils.jointpush_geometry import push_rsi_hand_targets, rotate, multiply
from utils.jointpush_rsi_geometry import align_push_pose, safe_place
from utils.jointpush_scene import sample_layout, place_distractors, support_clear, paths_clear, goal_angles_valid


class LegacyCache(SizeRsiCache):
    def __init__(self, task):
        original = task._box_size
        task._box_size = task._rsi_asset_sizes
        try:
            super().__init__(task)
        finally:
            task._box_size = original

    def _build_missing(self):
        self.eligible[self.env._push_assets] = 0
        super()._build_missing()


class LazyLegacyCache:
    def __init__(self, task):
        self.task = task
        self.cache = None

    def _load(self):
        if self.cache is None:
            self.cache = LegacyCache(self.task)
        return self.cache

    def availability(self, *args):
        return self._load().availability(*args)

    def sample(self, *args):
        return self._load().sample(*args)


class HumanoidMAPush(HumanoidMACarry):
    _solo_cache_class = LazyLegacyCache

    def __init__(self, cfg, *args, **kwargs):
        self._amp_num_families = 4
        self._joint = cfg['env']['jointPush']
        super().__init__(cfg, *args, **kwargs)

    def _motion_skills_to_load(self, available):
        return available

    def _load_box_asset(self):
        self._box_size, self._joint_probabilities = sample_assets(
            self.num_envs, self.cfg['env'], self.device, self._task_graph_preset)
        if (getattr(self.cfg['args'], 'eval_protocol', '') == 'stage1_single'
                and self.cfg['args'].eval_single_task == 'AT'):
            from utils.evaluation_protocol import single_at_sizes
            self._box_size[:, 0] = single_at_sizes(self.num_envs, self.device)
        self._push_assets = self._joint_probabilities[..., 5] > 0
        self._rsi_asset_sizes = self._box_size.clone()
        self._rsi_asset_sizes[:, :self.num_agents][self._push_assets] = .4
        self._box_scale = self._box_size/self._box_size.new_tensor(self._build_base_size)
        density = box_densities(self._box_size, self._push_assets, self._joint['box'])
        self._box_density = density.flatten()
        self._box_assets = []
        cache = {}
        for dims, rho in zip(self._box_size.flatten(0, 1).cpu().tolist(), self._box_density.cpu().tolist()):
            key = tuple(dims)+(rho,)
            if key not in cache:
                options = gymapi.AssetOptions()
                options.density = rho
                options.angular_damping = .01
                options.linear_damping = .01
                options.max_angular_velocity = 100.
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                cache[key] = self.gym.create_box(self.sim, *dims, options)
            self._box_assets.append(cache[key])

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        handle = self._box_handles[-1]
        if object_id < self.num_agents and self._push_assets[env_id, object_id]:
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for prop in props:
                prop.friction = self._joint['box']['friction']
                prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)
        state = self.gym.get_actor_rigid_body_states(env_ptr, handle, gymapi.STATE_ALL)
        state['pose']['p']['x'][:] = 5.*object_id
        state['pose']['p']['y'][:] = 8.
        self.gym.set_actor_rigid_body_states(env_ptr, handle, state, gymapi.STATE_ALL)

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        from utils.jointpush_rsi import PushBank
        self._bank = PushBank(self.cfg['env'], self._motion_lib)
        self._size_template_probabilities = self._joint_probabilities
        self.relation_runtime = Stage1PushRuntime(self.num_envs, self.relation_runtime.graph,
            self._relation_cfg, self.device, contact_weights=self._joint['contact_weights'])
        self._interaction = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool, device=self.device)
        self._push_hits = torch.zeros_like(self._interaction)
        self._push_goals = self._tar_pos.clone()
        self._anchor_roots = self._humanoid_root_states[..., :3].clone()

    def _templates(self, ids):
        env, agent = self._expand_slots(ids)
        return classify_templates(self.relation_runtime.graph, env, agent).reshape(len(ids), self.num_agents)

    def _sample_episode_graph(self, ids):
        weights = self._joint_probabilities[ids]
        templates = torch.multinomial(weights.flatten(0, 1), 1).reshape(len(ids), self.num_agents)
        graph = compose_graph(templates, self._relation_graph_spec['shuffle_edge_order'], self.num_objects)
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        for index in range(6):
            self._sampling_counts[index+1] += (templates == index).sum()
        for index, relation in enumerate((6, 7, 8, 9, 10, PUSH), 8):
            self._sampling_counts[index] += ((graph.edge_relation == relation) & graph.edge_valid).sum()

    def _consume_sampling_diagnostics(self):
        counts = self._sampling_counts.clone()
        push_rsi = self._size_rsi_executed[5].clone() if not self._is_eval else None
        result = super()._consume_sampling_diagnostics()
        denominator = (self.num_agents*counts[0]).clamp_min(1)
        for index, name in enumerate(('holding','sit','climb','holding_at','holding_ontop','push'), 1):
            result['sampling/template_'+name] = counts[index]/denominator
        result['sampling/shared_primary_object'] = counts.new_zeros(())
        for index, name in enumerate(('holding','at','ontop','sit','climb','push'), 8):
            result['sampling/relation_'+name] = counts[index]/denominator
        if push_rsi is not None:
            for index, name in enumerate(self._skill):
                result['sampling/rsi_push_actual_'+name] = push_rsi[index]/push_rsi.sum().clamp_min(1)
        return result

    def _record_reference(self, skill, env, owner, mids, times):
        previous = self._reset_ref_slots.get(skill)
        self._reset_ref_slots[skill] = (env, owner) if previous is None else tuple(
            torch.cat((p, v)) for p, v in zip(previous, (env, owner)))
        for mapping, value in ((self._reset_ref_motion_ids, mids), (self._reset_ref_motion_times, times)):
            mapping[skill] = value if skill not in mapping else torch.cat((mapping[skill], value))

    def _reset_ref_state_init(self, env, owner):
        push = classify_templates(self.relation_runtime.graph, env, owner) == 5
        if (~push).any():
            super()._reset_ref_state_init(env[~push], owner[~push])
        e, a = env[push], owner[push]
        if not len(e):
            return
        lib = self._motion_lib['push_loco']
        mids = lib.sample_motions(len(e))
        times = lib.sample_time_rsi(mids)
        root, q, dof, vel, ang, dof_vel, _ = lib.get_motion_state(mids, times)
        body, bq, bv, ba = lib.get_motion_state_max(mids, times)
        offset = self._agent_spawn_offsets[a]+self._env_origins[e]
        self._set_env_state(e, a, root+offset, q, dof, vel, ang, dof_vel)
        self._kinematic_humanoid_rigid_body_states[e, a] = torch.cat((body+offset[:, None], bq, bv, ba), -1)
        self._every_env_init_dof_pos[e, a] = dof
        probability = 0. if self._is_eval else self._joint['rsi']['interaction_probability']
        interaction = torch.rand(len(e), device=self.device) < probability
        self._interaction[e, a] = interaction
        if (~interaction).any():
            self._record_reference('loco', e[~interaction], a[~interaction], mids[~interaction], times[~interaction])
        if interaction.any():
            mids, times = self._bank.sample(int(interaction.sum()))
            self._record_reference('push', e[interaction], a[interaction], mids, times)

    def _reset_actors(self, ids):
        self._interaction[ids] = False
        self._push_hits[ids] = False
        super()._reset_actors(ids)
        self._anchor_roots[ids] = self._humanoid_root_states[ids, :, :3]

    def _reset_boxes(self, ids):
        super()._reset_boxes(ids)
        push = self._templates(ids) == 5
        boxes, goals = sample_layout(self._anchor_roots[ids], self._box_states[ids],
                                    self._box_size[ids], push, self._joint['scene'])
        self._box_states[ids] = boxes
        self._push_goals[ids] = goals
        slots = self._reset_ref_slots.get('push')
        if slots is None:
            return
        e, a = slots
        mids, times = self._reset_ref_motion_ids['push'], self._reset_ref_motion_times['push']
        lib = self._motion_lib['push']
        root, q, dof, vel, ang, dof_vel, _ = lib.get_motion_state(mids, times)
        body, bq, bv, ba = lib.get_motion_state_max(mids, times)
        cfg = dict(push=self._joint['contact'], contact_at=self._joint['contact'])
        targets = push_rsi_hand_targets(self._box_states[e, a, None], self._push_goals[e, a, None],
                                       self._box_size[e, a, None], cfg)[:, 0]
        body, bq, bv, ba, turn = align_push_pose(body, bq, bv, ba, targets,
            self._key_body_ids[:2].cpu().tolist(), self._joint['rsi']['stand_off'])
        body = safe_place(body, bq, self._box_states[e, a], self._box_size[e, a], targets, self._bank.geometry)
        self._set_env_state(e, a, body[:, 0], multiply(turn, q), dof, rotate(turn, vel), rotate(turn, ang), dof_vel)
        self._kinematic_humanoid_rigid_body_states[e, a] = torch.cat((body, bq, bv, ba), -1)
        self._every_env_init_dof_pos[e, a] = dof

    def _reset_task(self, ids):
        super()._reset_task(ids)
        rows, owners = (self._templates(ids) == 5).nonzero(as_tuple=True)
        self._tar_pos[ids[rows], owners] = self._push_goals[ids[rows], owners]

    def _configure_ontop_targets(self, ids):
        super()._configure_ontop_targets(ids)
        place_distractors(self, ids, self._templates(ids))

    def _ontop_scene_infeasible(self, ids):
        bad = super()._ontop_scene_infeasible(ids)
        push = self._templates(ids) == 5
        boxes, sizes, goals = self._box_states[ids], self._box_size[ids], self._tar_pos[ids]
        spec = self._joint['scene']
        bad |= ~support_clear(boxes, sizes, goals, push, spec).all(-1)
        bad |= ~paths_clear(boxes, sizes, goals, self._humanoid_root_states[ids, :, :3], push, spec)
        bad |= ~goal_angles_valid(boxes, goals, push, spec)
        distance = (goals[..., :2]-self._humanoid_root_states[ids, :, :2]).norm(dim=-1)
        bad |= ((distance < spec['goal_min_human_distance']) & push).any(-1)
        return bad

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        bodies = self._rigid_body_pos[ids] if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        hands = bodies[:, :, self._key_body_ids[:2]]
        from utils.jointpush_geometry import push_rear_xy
        distance = push_rear_xy(hands, self._box_states[ids, :self.num_agents], self._tar_pos[ids], self._box_size[ids, :self.num_agents]).amax(-1)
        root_distance = (bodies[:, :, 0, :2]-self._box_states[ids, :self.num_agents, :2]).norm(dim=-1)
        graph = self.relation_runtime.graph
        owner = graph.edge_owner[ids]
        mask = graph.edge_relation[ids] == PUSH
        spec = self._joint['contact']
        state = torch.exp(-spec['near_distance_scale']*distance.square())
        progress = 1/(1+(root_distance-spec['progress_delta']).clamp_min(0)/spec['progress_sigma'])
        phi = torch.where(mask, state.gather(1, owner), phi)
        for key, value in [('progress', progress), ('distance', distance), ('distance_xy', root_distance)]:
            diag[key] = torch.where(mask, value.gather(1, owner), diag[key])
        diag['z_error'] = diag['z_error'].masked_fill(mask, 0.)
        if env_ids is None and 'at_progress_min_delta' in self._joint:
            diag['progress'] = gate_push_at_progress(
                diag['progress'], diag['distance_xy'], self._prev_box_pos,
                self._tar_pos, graph, self._joint['at_progress_min_delta'])
        return phi, diag

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._push_hits.zero_()

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()
            _, hit = contact_state(self._rigid_body_pos[:, :, self._key_body_ids[:2]],
                self._box_states[:, :self.num_agents], self._tar_pos, self._box_size[:, :self.num_agents],
                self._contact_forces[:, :, self._key_body_ids[:2]], self._ontop_box_contact[:, :self.num_agents], self._joint['contact'])
            self._push_hits |= hit

    def _compute_reward(self, actions):
        self.relation_runtime.contact.copy_(self._push_hits)
        return super()._compute_reward(actions)

    def _amp_family(self, env_ids=None, slot_agent=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if slot_agent is None:
            env_ids, slot_agent = self._expand_slots(env_ids)
        template = classify_templates(self.relation_runtime.graph, env_ids, slot_agent)
        return torch.tensor([0, 1, 2, 0, 0, 3], device=self.device)[template]

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        if skill_name != 'loco':
            return super()._init_amp_obs_ref(slots, motion_ids, motion_times, skill_name)
        push = classify_templates(self.relation_runtime.graph, *slots) == 5
        for selected, name in ((~push, 'loco'), (push, 'push_loco')):
            if selected.any():
                super()._init_amp_obs_ref(tuple(v[selected] for v in slots), motion_ids[selected], motion_times[selected], name)

    def _fetch_conditioned_amp_demo(self, count):
        from utils.unified_training import append_family
        p = self._relation_graph_spec['template_probabilities']
        weights = torch.tensor([p['HOLDING_AT']+p['HOLDING_ON_TOP'], p['SIT'], p['CLIMB'], p['PUSH_AT']], device=self.device)
        family = torch.multinomial(weights, count, replacement=True)
        result = torch.empty(count, self.get_num_amp_obs(), device=self.device)
        for index, name in enumerate(('carry', 'sit', 'climb', 'push')):
            rows = (family == index).nonzero().flatten()
            if not len(rows):
                continue
            probs = torch.tensor(self.cfg['env']['ampExpertMapping'][name], device=self.device)
            selected = torch.multinomial(probs, len(rows), replacement=True)
            for skill, libname in enumerate(self.cfg['env']['skill']):
                dest = rows[selected == skill]
                if not len(dest):
                    continue
                if index == 3 and libname == 'loco':
                    libname = 'push_loco'
                lib = self._motion_lib[libname]
                mids = lib.sample_motions(len(dest))
                history = self.dt*(self._num_amp_obs_steps-1)
                times = lib.sample_time(mids, truncate_time=history)+history
                obs = self.build_amp_obs_demo(mids, times, lib).reshape(len(dest), self._num_amp_obs_steps, -1)
                result[dest] = append_family(obs, family[dest, None].expand(-1, self._num_amp_obs_steps), 4).flatten(1)
        return result
