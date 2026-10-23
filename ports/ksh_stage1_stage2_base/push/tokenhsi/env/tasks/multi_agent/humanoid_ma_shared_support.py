import torch
from isaacgym import gymapi

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from utils.shared_support import capacities, pack_tokens, partition, rotate, sample_groups, sample_templates
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import copy_graph_rows, permute_graph


class LocoRsi:
    def __init__(self, task):
        self.task = task

    def availability(self, rows, owners, templates):
        valid = torch.zeros(len(rows), len(self.task._skill), dtype=torch.bool, device=rows.device)
        valid[:, self.task._skill.index('loco')] = True
        return valid

    def sample(self, skill, rows, owners, templates):
        if skill != 'loco':
            raise ValueError('Shared support initialization is loco-only')
        lib = self.task._motion_lib['loco']
        clips = lib.sample_motions(len(rows))
        return clips, lib.sample_time_rsi(clips)


class HumanoidMASharedSupport(HumanoidMACarry):
    _solo_cache_class = LocoRsi

    def __init__(self, cfg, *args, **kwargs):
        cfg['env']['numObjects'], _, self._region_axes = capacities(cfg['env'])
        cfg['env']['relationGraph']['edge_capacity'] = 2*cfg['env']['numAgents']
        super().__init__(cfg, *args, **kwargs)

    def get_relation_suffix_size(self):
        return 6*self.num_agents

    def _load_box_asset(self):
        n, m, p = self.num_envs, self.num_agents, self.num_physical_objects
        self._box_spawn_radius = 4.5
        self._box_size = torch.empty(n, p, 3, device=self.device)
        ranges = self.cfg['env']['box']['build']['taskSizeRanges']
        specs = [(0, m, ranges['HOLDING_AT']), (m, 2*m, ranges['SIT']),
                 (2*m, 3*m, ranges['CLIMB']), (3*m, 4*m, ranges['SUPPORT']),
                 (4*m, 4*m+m//2, self.cfg['env']['sharedSupport']['size_ranges']),
                 (4*m+m//2, p, ranges['SUPPORT'])]
        for start, end, bounds in specs:
            for axis, (lo, hi) in enumerate(bounds):
                self._box_size[:, start:end, axis] = torch.randint(round(lo*20), round(hi*20)+1,
                    (n, end-start), device=self.device)/20
        self._box_scale = self._box_size / torch.tensor(self._build_base_size, device=self.device)
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
        n, m, o = self.num_envs, self.num_agents, self.num_objects
        self._groups = torch.arange(m, device=self.device)[None].expand(n, -1).clone()
        self._shared = torch.zeros(n, m, device=self.device, dtype=torch.bool)
        self._templates = torch.full((n, m), 3, device=self.device, dtype=torch.long)
        self._support_physical = torch.full_like(self._templates, -1)
        self._token_region = torch.full((n, o), -1, device=self.device, dtype=torch.long)
        self._task_token_count = torch.zeros(n, device=self.device, dtype=torch.long)
        self._region_offsets, self._region_sizes, self._region_valid = partition(
            self._box_size[:, 4*m:4*m+m//2], self._region_axes)

    def _logical_box_values(self, values, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        result = self._gather_entity_slots(values[ids], self._logical_box_order[ids]).clone()
        if self.num_agents < 2:
            return result
        region = self._token_region[ids]
        index = region.clamp_min(0)[..., None].expand(-1, -1, 3)
        if values is self._box_states:
            offsets = self._region_offsets[ids].flatten(1, 2).gather(1, index)
            offsets = offsets * (region >= 0)[..., None]
            delta = rotate(result[..., 3:7], offsets)
            result[..., :3] += delta
            result[..., 7:10] += torch.cross(result[..., 10:13], delta, dim=-1)
        elif values is self._box_size:
            cells = self._region_sizes[ids].flatten(1, 2).gather(1, index)
            result = torch.where((region >= 0)[..., None], cells, result)
        elif values is self._box_bps:
            signs = torch.tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],
                                  [1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]], device=self.device)
            result = self._logical_box_values(self._box_size, ids)[..., None, :]*signs/2
        return result

    def _sample_episode_graph(self, ids):
        m = self.num_agents
        spec = self.cfg['env']['sharedSupport']
        groups, shared = sample_groups(len(ids), m, spec['coupled_probability'], self.device, self._task_graph_preset)
        templates = sample_templates(len(ids), m, spec, shared, self.device, self._task_graph_preset)
        self._groups[ids], self._shared[ids], self._templates[ids] = groups, shared, templates
        people = torch.arange(m, device=self.device)[None].expand(len(ids), -1)
        source = people + torch.where(templates == 1, m, torch.where(templates == 2, 2*m, 0))
        leaders = shared & (groups == people)
        group_slot = (leaders.long().cumsum(-1)-1).gather(1, groups).clamp_min(0)
        support = torch.where(shared, 4*m+group_slot,
                              torch.where(templates == 4, 3*m+people, source))
        self._support_physical[ids] = torch.where(templates == 3, -1, support)
        self._agent_box_assignment[ids] = torch.where(shared & (templates < 3), support, source)
        group_active = torch.arange(m//2, device=self.device)[None] < leaders.sum(-1)[:, None]
        k = self._region_offsets.shape[2]
        parent = torch.arange(4*m, 4*m+m//2, device=self.device).repeat_interleave(k)
        parent = parent[None].expand(len(ids), -1)
        candidates = torch.cat((source, 3*m+people, parent), -1)
        valid = torch.cat((~shared | (templates == 4), ~shared & (templates == 4),
            (self._region_valid[ids] & group_active[..., None]).flatten(1)), -1)
        regions = torch.cat((torch.full_like(people.repeat(1, 2), -1),
            torch.arange(parent.shape[1], device=self.device)[None].expand(len(ids), -1)), -1)
        mapping, region, inverse, count = pack_tokens(candidates, valid, regions,
            self.num_objects, self.num_physical_objects, m)
        self._logical_box_order[ids], self._token_region[ids] = mapping, region
        self._task_token_count[ids] = count
        graph = compose_canonical_graph(templates, num_objects=self.num_objects)
        for endpoint in (graph.edge_src, graph.edge_dst):
            is_object = (endpoint >= m) & (endpoint < m+self.num_objects)
            old = (endpoint-m).clamp(0, 2*m-1)
            endpoint.copy_(torch.where(is_object, m+inverse.gather(1, old).clamp_max(self.num_objects-1), endpoint))
        if self._relation_graph_spec['shuffle_edge_order']:
            graph = permute_graph(graph, torch.rand(len(ids), 2*m, device=self.device).argsort(-1))
        copy_graph_rows(self.relation_runtime.graph, ids, graph)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:6] += torch.bincount(templates.flatten(), minlength=5)
        self._select_regions(ids, False)

    def _select_regions(self, ids, live):
        m = self.num_agents
        states = self._logical_box_values(self._box_states, ids)
        roots = self._rigid_body_pos[ids, :, 0, :3] if live else self._kinematic_humanoid_rigid_body_states[ids, :, 0, :3]
        point = torch.where((self._templates[ids] == 4)[..., None], self._box_states[ids, :m, :3], roots)
        parent = self._logical_box_order[ids]
        valid = (self._token_region[ids, None] >= 0) & (parent[:, None] == self._support_physical[ids, :, None])
        distance = (point[:, :, None, :2]-states[:, None, :, :2]).square().sum(-1)
        selected = distance.masked_fill(~valid, float('inf')).argmin(-1)+m
        graph = self.relation_runtime.graph
        owners = graph.edge_owner[ids]
        mask = self._shared[ids].gather(1, owners) & graph.edge_valid[ids] & (graph.edge_relation[ids] != 6)
        graph.edge_dst[ids] = torch.where(mask, selected.gather(1, owners), graph.edge_dst[ids])

    def _compute_clean_scene_obs(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._select_regions(ids, env_ids is None)
        raw = super()._compute_clean_scene_obs(env_ids)
        physical = torch.where(self._shared[ids], self._support_physical[ids], -1)
        return torch.cat((raw, physical.float()), -1)

    def _sample_box_assignments(self, ids):
        return

    def _active_physical(self, ids):
        active = torch.zeros(len(ids), self.num_physical_objects, dtype=torch.bool, device=self.device)
        active.scatter_(1, self._logical_box_order[ids], True)
        return active

    def _reset_all_boxes_random_arena(self, ids):
        super()._reset_all_boxes_random_arena(ids)
        inactive = ~self._active_physical(ids)
        es, box = inactive.nonzero(as_tuple=True)
        self._box_states[ids[es], box, :2] = self._env_origins[ids[es], :2]+1000+box[:, None]*3

    def _reset_boxes(self, ids):
        slots = self._reset_ref_slots['loco']
        solo = ~self._shared[slots]
        self._reset_ref_slots['loco'] = tuple(value[solo] for value in slots)
        super()._reset_boxes(ids)
        self._reset_ref_slots['loco'] = slots
        row, human = self._shared[ids].nonzero(as_tuple=True)
        es = ids[row]
        support = self._support_physical[es, human]
        box, size = self._box_states[es, support], self._box_size[es, support]
        side = torch.randint(4, (len(es),), device=self.device)
        axis = side//2
        along = torch.rand(len(es), device=self.device)*2-1
        spec = self.cfg['env']['sharedSupport']
        sit = torch.tensor(spec['spawn_distance_sit'], device=self.device)
        other = torch.tensor(spec['spawn_distance_other'], device=self.device)
        bounds = torch.where((self._templates[es, human] == 1)[:, None], sit, other)
        distance = bounds[:, 0]+torch.rand(len(es), device=self.device)*(bounds[:, 1]-bounds[:, 0])
        local = torch.zeros(len(es), 3, device=self.device)
        r = torch.arange(len(es), device=self.device)
        local[r, 1-axis] = along*size[r, 1-axis]/2
        local[r, axis] = (size[r, axis]/2+distance)*torch.where(side%2 == 0, -1., 1.)
        desired = box[:, :3]+rotate(box[:, 3:7], local)
        delta = desired[:, :2]-self._humanoid_root_states[es, human, :2]
        self._humanoid_root_states[es, human, :2] += delta
        self._kinematic_humanoid_rigid_body_states[es, human, :, :2] += delta[:, None]
        carry = self._templates[es, human] == 4
        from utils.stage1_sampling import sample_source_xy
        es, human = es[carry], human[carry]
        self._box_states[es, human, :2] = sample_source_xy(self._humanoid_root_states[es, human, :2], self._templates[es, human])

    def _configure_ontop_targets(self, ids):
        from utils.stage1_sampling import sample_goal_xy
        row, human = ((self._templates[ids] == 4) & ~self._shared[ids]).nonzero(as_tuple=True)
        es = ids[row]
        self._box_states[es, 3*self.num_agents+human, :2] = sample_goal_xy(
            self._humanoid_root_states[es, human, :2], self._box_states[es, human, :2], self._env_origins[es, :2])
        self._place_distractors(ids)

    def _place_distractors(self, ids):
        mapping = self._logical_box_order[ids]
        task_slots = torch.arange(self.num_objects, device=self.device)[None] < self._task_token_count[ids, None]
        occupied = torch.zeros(len(ids), self.num_physical_objects, dtype=torch.long, device=self.device)
        occupied.scatter_add_(1, mapping, task_slots.long())
        occupied = occupied > 0
        for slot in range(self.num_objects):
            rows = (~task_slots[:, slot]).nonzero().flatten()
            if not len(rows):
                continue
            es, physical = ids[rows], mapping[rows, slot]
            candidate = self._box_states[es, physical, :2].clone()
            sizes = self._box_size[es]
            clearance = sizes[..., :2].norm(dim=-1)/2 + self._box_size[es, physical, :2].norm(dim=-1)[:, None]/2 + .05
            for _ in range(64):
                distance = (candidate[:, None]-self._box_states[es, :, :2]).norm(dim=-1)
                blocked = ((distance < clearance) & occupied[rows]).any(-1)
                human_distance = (candidate[:, None]-self._humanoid_root_states[es, :, :2]).norm(dim=-1)
                blocked |= (human_distance < self._box_min_agent_dist).any(-1)
                if not blocked.any():
                    break
                candidate[blocked] = self._sample_arena_xy(self._env_origins[es[blocked], :2])
            self._box_states[es, physical, :2] = candidate
            occupied[rows, physical] = True

    def _ontop_scene_infeasible(self, ids):
        from env.tasks.multi_agent.shared_support_scene import scene_infeasible
        return scene_infeasible(self, ids)

    def _evaluate_ontop_context(self, env_ids=None):
        from env.tasks.multi_agent.shared_support_reward import evaluate_shared
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._select_regions(ids, env_ids is None)
        phi, diag = evaluate_shared(self, ids, env_ids is None)
        if env_ids is None:
            self._ontop_last_diag = diag
        return phi, diag

    def _video_focus_points(self):
        active = self._active_physical(torch.tensor([0], device=self.device))[0]
        return torch.cat((self._humanoid_root_states[0, :, :3], self._box_states[0, active, :3]), 0)

    def _update_camera(self):
        if getattr(self.cfg['args'], 'task_camera', 'stack') != 'stack' or not hasattr(self, '_shared'):
            return super()._update_camera()
        points = self._video_focus_points()
        low, high = points.amin(0), points.amax(0)
        center = (low+high)/2
        center[2] = max(float(center[2]), .8)
        radius = max(float((high-low)[:2].norm())*.8, 3.)
        if self._camera_follow_flag:
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(float(center[0])+radius*.65, float(center[1])-radius, 2.8+radius*.2),
                gymapi.Vec3(*center.tolist()))
        self._cam_prev_char_pos[:] = self._humanoid_root_states[0, 0, :3].cpu().numpy()
