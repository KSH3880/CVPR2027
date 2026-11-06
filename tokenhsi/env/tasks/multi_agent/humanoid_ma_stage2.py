import torch

from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
from env.tasks.multi_agent.before_reward import predecessor_at_progress
from env.tasks.multi_agent.stage2_layout import apply_before_layout, invalid_before_layout, resample_before_references
from env.tasks.multi_agent.stage2_reward import Stage2Runtime
from env.tasks.multi_agent.stage2_wait import Stage2Wait
from utils import coupled_push_spec as coupled
from utils import stage2_spec as spec
from utils.edge_ontop_spec import copy_graph_rows, select_graph
from env.tasks.multi_agent import stage2_group_layout as group_layout


class HumanoidMAStage2(HumanoidMACoupledPush):
    _resample_before_references = resample_before_references

    def _load_box_asset(self):
        ranges = self.cfg['env']['box']['build']['taskSizeRanges']
        original = ranges['HOLDING_AT']
        family = {'before_sit': 'SIT', 'before_climb': 'CLIMB'}.get(self._task_graph_preset)
        if family is not None:
            ranges['HOLDING_AT'] = [[max(a, c), min(b, d)]
                                   for (a, b), (c, d) in zip(original, ranges[family])]
        try:
            super()._load_box_asset()
        finally:
            ranges['HOLDING_AT'] = original

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        self._templates = self._templates_by_owner
        self._before_predecessor = torch.full_like(self._groups, -1)
        self._before_near_pairs = torch.zeros_like(self._groups, dtype=torch.bool)
        self._stage2_task_slots = torch.zeros(self.num_envs, self.num_objects,
                                             dtype=torch.bool, device=self.device)
        self._scene_domains = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stage2_domain_counts = torch.zeros(3, device=self.device)
        self._group_spawn_targets = torch.zeros(self.num_envs, self.num_agents, 2, device=self.device)

    def _reset_actors(self, ids):
        super()._reset_actors(ids)
        if self.num_agents > 4:
            group_layout.translate_people(self, ids)

    def _compute_clean_scene_obs(self, env_ids=None):
        obs = super()._compute_clean_scene_obs(env_ids)
        if self.num_agents > 4:
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            obs = group_layout.clean_observation(self, obs, ids)
        return obs

    def _update_camera(self):
        if self.num_agents > 4 and getattr(self.cfg['args'], 'task_camera', 'stack') == 'stack':
            return group_layout.camera(self)
        return super()._update_camera()

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self.relation_runtime = Stage2Runtime(self.num_envs, self.relation_runtime.graph,
            self._relation_cfg, self.device, contact_weights=self._joint['contact_weights'],
            before_at_z_tolerance=self.cfg['env']['stage2']['before_at_z_tolerance'])
        self._before_wait = Stage2Wait(self.relation_runtime.graph, self.num_envs,
                                       self.cfg['env']['before']['waiting_penalty'], self.device)

    def _sample_episode_graph(self, ids):
        domain, groups, predecessor, templates = spec.sample_episode(
            self._box_size[ids, :self.num_agents], self.cfg['env']['box']['build']['taskSizeRanges'],
            self._relation_graph_spec['template_probabilities'], self._task_graph_preset,
            self.cfg['env']['stage2']['scene_weights'], num_objects=self.num_objects)
        graph = spec.compose_graph(templates, groups, predecessor, self.num_objects,
                                   self._relation_graph_spec['shuffle_edge_order'])
        mapping, primary, slots = spec.physical_mapping(templates, groups, predecessor, graph)
        self._groups[ids], self._templates[ids] = groups, templates
        self._before_predecessor[ids] = predecessor
        self._before_near_pairs[ids] = (predecessor >= 0) & (torch.rand_like(predecessor, dtype=torch.float) < .5)
        self._members[ids] = torch.where(templates == 5, coupled.group_sizes(groups), 1)
        self._regions[ids] = torch.where(self._members[ids] > 1, coupled.assign_regions(groups), 0)
        self._agent_box_assignment[ids], self._logical_box_order[ids] = primary, mapping
        self._stage2_task_slots[ids], self._scene_domains[ids] = slots, domain
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self.relation_runtime.before[ids] = spec.before_members(predecessor)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:7] += torch.bincount(templates.flatten(), minlength=6)
        self._stage2_domain_counts += torch.bincount(domain, minlength=3)
        self._push_hits[ids] = False
        self._update_box_assignment_colors(ids)

    def _reset_boxes(self, ids):
        slots = self._reset_ref_slots.get('loco')
        if slots is not None:
            env, owner = slots
            keep = (self._before_predecessor[env, owner] < 0) | (self._templates[env, owner] == 4)
            self._reset_ref_slots['loco'] = (env[keep], owner[keep])
        super()._reset_boxes(ids)
        if slots is not None:
            self._reset_ref_slots['loco'] = slots

    def _configure_ontop_targets(self, ids):
        slots = self._reset_ref_slots.get('loco')
        if slots is not None:
            env, owner = slots
            keep = self._before_predecessor[env, owner] < 0
            self._reset_ref_slots['loco'] = (env[keep], owner[keep])
        super()._configure_ontop_targets(ids)
        if slots is not None:
            self._reset_ref_slots['loco'] = slots
        apply_before_layout(self, ids)
        if self.num_agents > 4:
            group_layout.recenter_before(self, ids)

    def _ontop_scene_infeasible(self, ids):
        bad = super()._ontop_scene_infeasible(ids) | invalid_before_layout(self, ids)
        roots = self._humanoid_root_states[ids, :, :2]
        gap = (roots[:, :, None]-roots[:, None]).norm(dim=-1)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
        before_scene = self._scene_domains[ids] != 1
        return bad | (before_scene & ((gap < .7) & ~eye).any((1, 2)))

    def _evaluate_ontop_context(self, env_ids=None):
        phi, diag = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.relation_runtime.gate[ids] |= self._members[ids] == 1
        bodies = self._rigid_body_pos[ids] if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids, ..., :3]
        graph = self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph, ids)
        diag['progress'] = predecessor_at_progress(diag['progress'], bodies[..., 0, :],
            self._logical_box_values(self._box_states, env_ids), self._logical_box_values(self._box_size, env_ids),
            self._tar_pos[ids], graph, self._relation_cfg)
        return phi, diag

    def _reset_relation_history(self, ids):
        super()._reset_relation_history(ids)
        self._before_wait.reset(ids, self.relation_runtime.graph, self.relation_runtime.own_success)

    def _fetch_conditioned_amp_demo(self, count):
        self._stage2_amp_weights = torch.bincount(self._amp_family(), minlength=4).float()
        return super()._fetch_conditioned_amp_demo(count)

    def _consume_sampling_diagnostics(self):
        result = super()._consume_sampling_diagnostics()
        total = self._stage2_domain_counts.sum().clamp_min(1)
        for index, name in enumerate(('before', 'coupled', 'mixed')):
            result['sampling/stage2_'+name] = self._stage2_domain_counts[index]/total
        self._stage2_domain_counts.zero_()
        return result
