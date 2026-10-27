from dataclasses import replace
from utils.before_full_rsi import full_rsi
from env.tasks.multi_agent import before_full_rsi as full_reset

from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_shared_support import HumanoidMASharedSupport
from utils.before_spec import sample_graph, pack_objects, task_objects
from utils.edge_ontop_spec import copy_graph_rows, permute_graph, select_graph
from utils.stage1_sampling import sample_goal_xy
from utils.before_sampling import (sample_distance_modes, sample_follower_modes,
    shorten_leader_xy, sample_before_goal_xy)
from env.tasks.multi_agent.before_carry_rsi import make_cache, BeforeCarryRsiCache


class HumanoidMABefore(HumanoidMACarry):
    _solo_cache_class = staticmethod(make_cache)
    _active_physical = HumanoidMASharedSupport._active_physical
    _place_distractors = HumanoidMASharedSupport._place_distractors
    def _ontop_scene_infeasible(self, ids):
        base = lambda rows: HumanoidMASharedSupport._ontop_scene_infeasible(self, rows)
        return full_reset.screen(self, ids, base) if full_rsi(self.cfg['env']) else base(ids)

    def _load_box_asset(self):
        n, m, p = self.num_envs, self.num_agents, self.num_physical_objects
        self._box_spawn_radius = 4.5
        self._box_size = torch.empty(n, p, 3, device=self.device)
        ranges = self.cfg['env']['box']['build']['taskSizeRanges']
        for bank, name in enumerate(('HOLDING_AT', 'SIT', 'CLIMB', 'SUPPORT')):
            bounds = [list(axis) for axis in ranges[name]]
            family = self._task_graph_preset.rsplit('_', 1)[-1].upper()
            if bank == 0 and family in ('SIT', 'CLIMB') and self._task_graph_preset.startswith(('before_', 'one_pair_')):
                bounds = [[max(lo, a), min(hi, b)] for (lo, hi), (a, b) in zip(bounds, ranges[family])]
            for axis, (lo, hi) in enumerate(bounds):
                self._box_size[:, bank*m:(bank+1)*m, axis] = torch.randint(
                    round(lo*20), round(hi*20)+1, (n, m), device=self.device)/20
        for axis, (lo, hi) in enumerate(ranges['SUPPORT']):
            self._box_size[:, 4*m:, axis] = torch.randint(
                round(lo*20), round(hi*20)+1, (n, p-4*m), device=self.device)/20
        self._box_scale = self._box_size/torch.tensor(self._build_base_size, device=self.device)
        self._box_density = torch.full((n*p,), 100., device=self.device)
        self._box_assets = []
        for size in self._box_size.reshape(-1, 3).cpu().tolist():
            options = gymapi.AssetOptions()
            options.angular_damping = options.linear_damping = .01
            options.max_angular_velocity = 100.
            options.density = 100.
            options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, *size, options))

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        if full_rsi(self.cfg['env']):
            full_reset.allocate(self)
        shape = (self.num_envs, self.num_agents)
        self._templates = torch.full(shape, 3, device=self.device, dtype=torch.long)
        self._before_predecessor = torch.full_like(self._templates, -1)
        self._before_near_pairs = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self._before_distance_modes = torch.zeros_like(self._templates)
        self._before_mode_counts = torch.zeros(3, device=self.device)
        self._before_mixture_counts = torch.zeros(2, device=self.device)
        self._task_token_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._before_carry_mask = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self._before_carry_counts = torch.zeros(4, device=self.device)

    def _init_ontop_context_runtime(self):
        super()._init_ontop_context_runtime()
        from env.tasks.multi_agent.before_reward import BeforeRuntime
        self.relation_runtime = BeforeRuntime(self.num_envs, self.relation_runtime.graph,
                                              self._relation_cfg, self.device)

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        if self.cfg['env']['before'].get('progress_target') == 'predecessor_at_goal':
            from env.tasks.multi_agent.before_reward import predecessor_at_progress
            bodies = (self._rigid_body_pos if env_ids is None
                      else self._kinematic_humanoid_rigid_body_states[env_ids, ..., :3])
            graph = (self.relation_runtime.graph if env_ids is None
                     else select_graph(self.relation_runtime.graph, env_ids))
            goals = self._tar_pos if env_ids is None else self._tar_pos[env_ids]
            diag['progress'] = predecessor_at_progress(diag['progress'], bodies[..., 0, :],
                self._logical_box_values(self._box_states, env_ids),
                self._logical_box_values(self._box_size, env_ids), goals, graph, self._relation_cfg)
        return phi, diag

    def _sample_episode_graph(self, ids):
        m = self.num_agents
        graph, predecessor, templates = sample_graph(self._box_size[ids, :m],
            self.cfg['env']['box']['build']['taskSizeRanges'], self._task_graph_preset, self.num_objects)
        self._before_predecessor[ids], self._templates[ids] = predecessor, templates
        # Draw once per episode graph; physical placement retries retain the mode.
        mixture = self.cfg['env']['before'].get('distance_mixture')
        force_near = self._is_eval and getattr(self.cfg['args'], 'before_follower_near', False)
        force_short = self._is_eval and getattr(self.cfg['args'], 'before_follower_short', False)
        modes = sample_distance_modes(predecessor, mixture, force_near, force_short)
        self._before_distance_modes[ids] = modes
        self._before_near_pairs[ids] = modes > 0
        if mixture is not None:
            self._before_mode_counts += torch.bincount(modes[predecessor >= 0], minlength=3)
            self._before_mixture_counts[0] += self._before_near_pairs[ids].sum()
            self._before_mixture_counts[1] += (predecessor >= 0).sum()
        people = torch.arange(m, device=self.device)[None].expand(len(ids), -1)
        source = people+torch.where(templates == 1, m, torch.where(templates == 2, 2*m, 0))
        follower = predecessor >= 0
        self._agent_box_assignment[ids] = torch.where(follower & (templates < 3), predecessor, source)
        candidates = torch.cat((source, 3*m+people), -1)
        valid = task_objects(predecessor, templates)
        mapping, inverse, count = pack_objects(candidates, valid, self.num_objects)
        self._logical_box_order[ids], self._task_token_count[ids] = mapping, count
        for endpoint in (graph.edge_src, graph.edge_dst):
            objects = (endpoint >= m) & (endpoint < 3*m)
            goals = endpoint >= 3*m
            old = (endpoint-m).clamp(0, 2*m-1)
            endpoint.copy_(torch.where(objects, m+inverse.gather(1, old),
                                      torch.where(goals, endpoint+self.num_objects-2*m, endpoint)))
        graph = replace(graph, num_objects=self.num_objects)
        if self._relation_graph_spec['shuffle_edge_order']:
            graph = permute_graph(graph, torch.rand(len(ids), 2*m, device=self.device).argsort(-1))
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._update_box_assignment_colors(ids)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:6] += torch.bincount(templates.flatten(), minlength=5)
        if full_rsi(self.cfg['env']):
            full_reset.new_graph(self, ids)
        spec = self.cfg['env']['before'].get('follower_carry_rsi')
        if spec is not None:
            mode = getattr(self.cfg['args'], 'before_carry_rsi', 'loco') if self._is_eval else 'mix'
            probability = {'loco': 0., 'mix': spec['probability'], 'carry': 1.}[mode]
            available = torch.zeros_like(templates, dtype=torch.bool)
            if probability:
                if not hasattr(self, '_size_rsi_cache'):
                    self._size_rsi_cache = BeforeCarryRsiCache(self)
                available = self._size_rsi_cache.carry_available[ids]
            from utils.before_carry_rsi import select_carry
            eligible, requested, selected = select_carry(predecessor, templates, probability, available)
            self._before_carry_mask[ids] = selected
            self._before_carry_counts += torch.stack((eligible.sum(), requested.sum(),
                                                     selected.sum(), (requested & ~selected).sum()))

    def _reset_ref_state_init(self, slot_env, slot_agent):
        if full_rsi(self.cfg['env']):
            return full_reset.reset_ref(self, slot_env, slot_agent)
        carry = self._before_carry_mask[slot_env, slot_agent]
        if not carry.any():
            return super()._reset_ref_state_init(slot_env, slot_agent)
        if (~carry).any():
            super()._reset_ref_state_init(slot_env[~carry], slot_agent[~carry])
        rows, owners = slot_env[carry], slot_agent[carry]
        clips, times = self._size_rsi_cache.sample('carryWith', rows, owners, self._templates[rows, owners])
        lib = self._motion_lib['carryWith']
        root, rot, dofs, vel, ang, dof_vel, _ = lib.get_motion_state(clips, times)
        offset = self._agent_spawn_offsets[owners] + self._env_origins[rows]
        self._set_env_state(rows, owners, root_pos=root+offset, root_rot=rot,
            dof_pos=dofs, root_vel=vel, root_ang_vel=ang, dof_vel=dof_vel)
        pos, quat, linear, angular = lib.get_motion_state_max(clips, times)
        self._kinematic_humanoid_rigid_body_states[rows, owners] = torch.cat(
            (pos+offset[:, None], quat, linear, angular), -1)
        self._every_env_init_dof_pos[rows, owners] = dofs
        self._reset_ref_slots['carryWith'] = (rows, owners)
        self._reset_ref_motion_ids['carryWith'] = clips
        self._reset_ref_motion_times['carryWith'] = times
        if not self._is_eval:
            self._size_rsi_requested[4, self._skill.index('carryWith')] += len(rows)

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)
        if full_rsi(self.cfg['env']):
            for skill, slots in self._reset_ref_slots.items():
                if skill not in self._skill:
                    self._init_amp_obs_ref(slots, self._reset_ref_motion_ids[skill],
                                           self._reset_ref_motion_times[skill], skill)
            return
        if 'carryWith' not in self._skill and 'carryWith' in self._reset_ref_slots:
            self._init_amp_obs_ref(self._reset_ref_slots['carryWith'],
                self._reset_ref_motion_ids['carryWith'],
                self._reset_ref_motion_times['carryWith'], 'carryWith')

    def _sample_box_assignments(self, ids):
        return

    def _reset_all_boxes_random_arena(self, ids):
        super()._reset_all_boxes_random_arena(ids)
        row, box = (~self._active_physical(ids)).nonzero(as_tuple=True)
        self._box_states[ids[row], box, :2] = self._env_origins[ids[row], :2]+1000+box[:, None]*3

    def _reset_boxes(self, ids):
        if full_rsi(self.cfg['env']):
            return full_reset.reset_boxes(self, ids, super()._reset_boxes)
        slots = self._reset_ref_slots.get('loco')
        if slots is not None:
            own_payload = (self._before_predecessor[slots] < 0) | (self._templates[slots] == 4)
            self._reset_ref_slots['loco'] = tuple(value[own_payload] for value in slots)
        super()._reset_boxes(ids)
        if slots is not None:
            self._reset_ref_slots['loco'] = slots
        row, human = (self._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
        es = ids[row]
        first = self._before_predecessor[es, human]
        modes = self._before_distance_modes[es, human]
        mixture = self.cfg['env']['before'].get('distance_mixture')
        short = modes == 2
        se, leader = es[short], first[short]
        leader_xy = self._humanoid_root_states[se, leader, :2]
        delta = shorten_leader_xy(leader_xy, self._box_states[se, leader, :2]) - leader_xy
        self._humanoid_root_states[se, leader, :2] += delta
        self._kinematic_humanoid_rigid_body_states[se, leader, :, :2] += delta[:, None]
        desired = sample_follower_modes(self._box_states[es, first, :2], self._templates[es, human], modes, mixture)
        delta = desired-self._humanoid_root_states[es, human, :2]
        self._humanoid_root_states[es, human, :2] += delta
        self._kinematic_humanoid_rigid_body_states[es, human, :, :2] += delta[:, None]
        top = self._templates[es, human] == 4
        carried = self._before_carry_mask[es, human]
        self._box_states[es[carried], human[carried], :2] += delta[carried]
        if carried.any():
            from utils.before_layout import transform
            from utils.shared_support import rotate
            ce, ch = es[carried], human[carried]
            angle = torch.rand(len(ce), device=self.device)*2*torch.pi
            q = torch.zeros(len(ce), 4, device=self.device)
            q[:, 2], q[:, 3] = torch.sin(angle/2), torch.cos(angle/2)
            root = self._humanoid_root_states[ce, ch]
            shift = root[:, :3]-rotate(q, root[:, :3])
            self._humanoid_root_states[ce, ch] = transform(root, q, shift)
            self._kinematic_humanoid_rigid_body_states[ce, ch] = transform(
                self._kinematic_humanoid_rigid_body_states[ce, ch], q, shift)
            self._box_states[ce, ch] = transform(self._box_states[ce, ch], q, shift)
        ground_top = top & ~carried
        self._box_states[es[ground_top], human[ground_top], :2] = sample_follower_modes(
            desired[ground_top], self._templates[es[ground_top], human[ground_top]], modes[ground_top], mixture, payload=True)
        if (getattr(self.cfg['args'], 'before_follower_near', False)
                and self.cfg['env']['before'].get('distance_mixture') is None):
            from env.tasks.multi_agent.before_follower_near import apply_layout
            apply_layout(self, ids)

    def _reset_task(self, ids):
        if full_rsi(self.cfg['env']):
            return full_reset.reset_goals(self, ids, super()._reset_task)
        super()._reset_task(ids)
        mixture = self.cfg['env']['before'].get('distance_mixture')
        if mixture is None or 'leader_goal_distance' not in mixture:
            return
        row, follower = (self._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
        es = ids[row]
        if not len(es):
            return
        leader = self._before_predecessor[es, follower]
        payload = self._scenario_physical_object(es, leader)
        goal = self._scenario_goal(es, leader)
        self._tar_pos[es, goal, :2] = sample_before_goal_xy(
            self._humanoid_root_states[es, leader, :2], self._box_states[es, payload, :2],
            self._before_distance_modes[es, follower], mixture)

    def _configure_ontop_targets(self, ids):
        if full_rsi(self.cfg['env']):
            return full_reset.ontop_targets(self, ids)
        row, human = ((self._templates[ids] == 4) & (self._before_predecessor[ids] < 0)).nonzero(as_tuple=True)
        es = ids[row]
        self._box_states[es, 3*self.num_agents+human, :2] = sample_goal_xy(
            self._humanoid_root_states[es, human, :2], self._box_states[es, human, :2], self._env_origins[es, :2])
        self._place_distractors(ids)

    def _record_ontop_diagnostics(self, result, diag):
        super()._record_ontop_diagnostics(result, diag)
        dependent = self.relation_runtime.graph.prereq_mask.any(-1)
        for name, values in (('gate', result['prerequisite_gate']),
                             ('raw_success', result['raw_own_success']),
                             ('success', result['own_success'])):
            key = 'before/'+name
            self._edge_metric_sums[key] = self._edge_metric_sums.get(key, 0)+(values & dependent).sum()
            self._edge_metric_denominators[key] = self._edge_metric_denominators.get(key, 0)+dependent.sum()
            if self.cfg['env']['before'].get('progress_target') == 'predecessor_at_goal':
                for relation, skill in ((9, 'sit'), (10, 'climb'), (8, 'ontop')):
                    mask = dependent & (self.relation_runtime.graph.edge_relation == relation)
                    split_key = 'before/'+skill+'/'+name
                    self._edge_metric_sums[split_key] = self._edge_metric_sums.get(split_key, 0)+(values & mask).sum()
                    self._edge_metric_denominators[split_key] = self._edge_metric_denominators.get(split_key, 0)+mask.sum()

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        if full_rsi(self.cfg['env']):
            result.update(full_reset.diagnostics(self))
        if self.cfg['env']['before'].get('follower_carry_rsi') is not None:
            eligible, requested, executed, fallback = self._before_carry_counts
            for key, value in (('eligible', eligible), ('requested', requested),
                               ('executed', executed), ('fallback', fallback)):
                result['sampling/before_carry_'+key] = value.clone()
            result['sampling/before_carry_fraction'] = executed/eligible.clamp_min(1)
            self._before_carry_counts.zero_()
        if self.cfg['env']['before'].get('distance_mixture') is not None:
            near, total = self._before_mixture_counts
            result['sampling/before_near_fraction'] = near / total.clamp_min(1)
            result['sampling/before_pairs'] = total.clone()
            if 'probabilities' in self.cfg['env']['before']['distance_mixture']:
                for index, name in enumerate(('original', 'medium', 'short')):
                    result['sampling/before_'+name+'_fraction'] = self._before_mode_counts[index] / total.clamp_min(1)
            self._before_mode_counts.zero_()
            self._before_mixture_counts.zero_()
        return result
