from dataclasses import replace

from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi, HumanoidMASharedSupport
from utils.before_spec import sample_graph, pack_objects, task_objects
from utils.edge_ontop_spec import copy_graph_rows, permute_graph
from utils.stage1_sampling import sample_goal_xy
from utils.before_sampling import sample_distance_modes, sample_follower_modes, shorten_leader_xy


class HumanoidMABefore(HumanoidMACarry):
    _solo_cache_class = LocoRsi
    _active_physical = HumanoidMASharedSupport._active_physical
    _place_distractors = HumanoidMASharedSupport._place_distractors
    _ontop_scene_infeasible = HumanoidMASharedSupport._ontop_scene_infeasible

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
        shape = (self.num_envs, self.num_agents)
        self._templates = torch.full(shape, 3, device=self.device, dtype=torch.long)
        self._before_predecessor = torch.full_like(self._templates, -1)
        self._before_near_pairs = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self._before_distance_modes = torch.zeros_like(self._templates)
        self._before_mode_counts = torch.zeros(3, device=self.device)
        self._before_mixture_counts = torch.zeros(2, device=self.device)
        self._task_token_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _init_ontop_context_runtime(self):
        super()._init_ontop_context_runtime()
        from env.tasks.multi_agent.before_reward import BeforeRuntime
        self.relation_runtime = BeforeRuntime(self.num_envs, self.relation_runtime.graph,
                                              self._relation_cfg, self.device)

    def _sample_episode_graph(self, ids):
        m = self.num_agents
        graph, predecessor, templates = sample_graph(self._box_size[ids, :m],
            self.cfg['env']['box']['build']['taskSizeRanges'], self._task_graph_preset, self.num_objects)
        self._before_predecessor[ids], self._templates[ids] = predecessor, templates
        # Draw once per episode graph; physical placement retries retain the mode.
        mixture = self.cfg['env']['before'].get('distance_mixture')
        force_near = self._is_eval and getattr(self.cfg['args'], 'before_follower_near', False)
        modes = sample_distance_modes(predecessor, mixture, force_near)
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

    def _sample_box_assignments(self, ids):
        return

    def _reset_all_boxes_random_arena(self, ids):
        super()._reset_all_boxes_random_arena(ids)
        row, box = (~self._active_physical(ids)).nonzero(as_tuple=True)
        self._box_states[ids[row], box, :2] = self._env_origins[ids[row], :2]+1000+box[:, None]*3

    def _reset_boxes(self, ids):
        slots = self._reset_ref_slots['loco']
        own_payload = (self._before_predecessor[slots] < 0) | (self._templates[slots] == 4)
        self._reset_ref_slots['loco'] = tuple(value[own_payload] for value in slots)
        super()._reset_boxes(ids)
        self._reset_ref_slots['loco'] = slots
        row, human = (self._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
        es = ids[row]
        first = self._before_predecessor[es, human]
        modes = self._before_distance_modes[es, human]
        short = modes == 2
        se, leader = es[short], first[short]
        leader_xy = self._humanoid_root_states[se, leader, :2]
        delta = shorten_leader_xy(leader_xy, self._box_states[se, leader, :2]) - leader_xy
        self._humanoid_root_states[se, leader, :2] += delta
        self._kinematic_humanoid_rigid_body_states[se, leader, :, :2] += delta[:, None]
        desired = sample_follower_modes(self._box_states[es, first, :2], self._templates[es, human], modes)
        delta = desired-self._humanoid_root_states[es, human, :2]
        self._humanoid_root_states[es, human, :2] += delta
        self._kinematic_humanoid_rigid_body_states[es, human, :, :2] += delta[:, None]
        top = self._templates[es, human] == 4
        self._box_states[es[top], human[top], :2] = sample_follower_modes(
            desired[top], self._templates[es[top], human[top]], modes[top])
        if (getattr(self.cfg['args'], 'before_follower_near', False)
                and self.cfg['env']['before'].get('distance_mixture') is None):
            from env.tasks.multi_agent.before_follower_near import apply_layout
            apply_layout(self, ids)

    def _configure_ontop_targets(self, ids):
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

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
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
