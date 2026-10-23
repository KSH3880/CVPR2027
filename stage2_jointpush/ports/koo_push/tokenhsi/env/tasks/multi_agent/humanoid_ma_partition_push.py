"""Four humans in variable solo/coupled Push groups, plus four random distractors."""
from types import SimpleNamespace
from pathlib import Path
import hashlib
from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.humanoid_ma_joint_push import cpa_penalty
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.partition_push import nearest_boxes, group_tokens, group_layout, sample_groups, sample_asset_bank, partition_reward, sample_surround_spawn, exclusive_regions, hidden_door_region_tokens, sample_virtual_doors
from utils.open_push_spec import compile_graph
from utils.edge_ontop_spec import expand_graph
from utils.edge_stage1_spec import semantic_graph_packet


class HumanoidMAPartitionPush(HumanoidMACarry):
    REGION_ASSIGNMENT = 'exclusive_reset_min_distance_v1'
    REWARD_TERM_NAMES = ('state', 'progress', 'success', 'power', 'collision', 'total')

    def __init__(self, cfg, *args, **kwargs):
        self._partition_push = self._joint_push = True
        self._joint_cfg = cfg['env']
        self._joint_push_spec = cfg['env']['partitionPush']
        layout = self._joint_push_spec.get('object_layout', 'regions_and_boxes')
        if layout not in ('regions_and_boxes', 'hidden_doors_then_regions_v1', 'hidden_doors_scattered_regions_v2', 'task_regions_only_v1'):
            raise ValueError('Unknown Partition Push object layout: '+layout)
        self._task_only = layout == 'task_regions_only_v1'
        self._bank_size = 16 if self._task_only else 24
        self._hidden_door_tokens = layout in ('hidden_doors_then_regions_v1', 'hidden_doors_scattered_regions_v2')
        self._scattered_door_tokens = layout == 'hidden_doors_scattered_regions_v2'
        self._amp_family_count = 2
        if (cfg['env']['numAgents'], cfg['env']['numObjects']) != (4, 4 if self._task_only else 8):
            raise ValueError('Partition Push requires 4 humans and 8 object tokens')
        if cfg['env'].get('openPushRSI') or cfg['env']['skillInitProb'] != [1., 0., 0.]:
            raise ValueError('Partition Push requires loco-only initialization')
        super().__init__(cfg, *args, **kwargs)
        self._amp_expert_skills = list(cfg['env']['skill'])
        self._env_origins.zero_()

    def _load_box_asset(self):
        spec = self._joint_push_spec
        self._box_size, self._payload_masses = sample_asset_bank(self.num_envs, spec, self.device, task_only=self._task_only)
        self._box_assets = []
        for size, mass in zip(self._box_size.flatten(0, 1).cpu().tolist(), self._payload_masses.flatten().cpu().tolist()):
            options = gymapi.AssetOptions()
            options.density = mass / (size[0]*size[1]*size[2])
            options.angular_damping = options.linear_damping = .01
            self._box_assets.append(self.gym.create_box(self.sim, *size, options))

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        HumanoidMA._build_env(self, env_id, env_ptr, humanoid_asset)
        for box_id in range(self._bank_size):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0., 0.,
                                 float(self._box_size[env_id, box_id, 2])/2)
            handle = self.gym.create_actor(env_ptr, self._box_assets[env_id*self._bank_size+box_id], pose,
                                           'push_box_'+str(box_id), env_id, 0, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for prop in props:
                prop.friction = self.cfg['env']['pushBox']['friction']; prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)
            self._box_handles.append(handle)
        if self._enable_markers:
            for agent in range(self.num_agents):
                self._build_marker(env_id, agent, env_ptr)

    def _build_box_tensors(self):
        actors = self.get_num_actors_per_env()
        self._box_states = self._root_states.view(self.num_envs, actors, 13)[:, 4:4+self._bank_size]
        self._box_actor_ids = (actors*torch.arange(self.num_envs, device=self.device, dtype=torch.int32)[:, None]
                               + torch.arange(self._bank_size, device=self.device, dtype=torch.int32)[None] + 4)
        self._initial_box_states = self._box_states.clone()
        self._build_box_bps()

    def _build_marker_state_tensors(self):
        actors = self.get_num_actors_per_env()
        self._marker_states = self._root_states.view(self.num_envs, actors, 13)[:, 4+self._bank_size:8+self._bank_size]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_actor_ids = (actors*torch.arange(self.num_envs, device=self.device, dtype=torch.int32)[:, None]
                                  + torch.arange(4+self._bank_size, 8+self._bank_size, device=self.device, dtype=torch.int32)[None])

    def get_clean_object_obs_size(self):
        return 40  # Preserve the Stage1 tokenizer; articulation fields are zero.

    def _clean_object_features(self, object_local, env_ids):
        extra = object_local.new_zeros(*object_local.shape[:-1], 10)
        if getattr(self, '_hidden_door_tokens', False):
            extra[:, :4, 2:5] = extra.new_tensor([.78, -.10, 1.])
            extra[:, :4, 8] = 1.  # Existing Stage1 fixed-door articulation flag.
            # Joint Push leaves push_eligible at its reset value (zero).
        return torch.cat((object_local, extra), -1)

    def _logical_box_values(self, values, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        states, sizes, _ = group_tokens(self._box_states[ids], self._box_size[ids],
            self._agent_box_assignment[ids], self._ordinary_keep[ids])
        if getattr(self, '_hidden_door_tokens', False):
            doors = self._virtual_door_states[ids] if getattr(self, '_scattered_door_tokens', False) else None
            states, sizes = hidden_door_region_tokens(states, sizes, doors)
        if values is self._box_states: return states
        if values is self._box_size: return sizes
        if values is self._box_bps:
            signs = self._box_bps.new_tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],
                                             [1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]])
            corners = sizes[..., None, :]*signs/2
            if getattr(self, '_hidden_door_tokens', False):
                corners[:, :4] += corners.new_tensor([.45, 0., 1.02])
            return corners
        raise ValueError('Unknown partitioned object field')

    def _init_relation_runtime(self):
        graph = expand_graph(compile_graph(self._relation_graph_spec, 4, self.num_objects, self.device), self.num_envs)
        runtime = SimpleNamespace(graph=graph, tasks=torch.ones(self.num_envs, 4, device=self.device, dtype=torch.long))
        for key in ('phi', 'error'):
            setattr(runtime, key, torch.zeros(self.num_envs, 8, device=self.device))
        for key in ('own_success', 'achieved'):
            setattr(runtime, key, torch.zeros(self.num_envs, 8, device=self.device, dtype=torch.bool))
        runtime.done = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        runtime.suffix = lambda ids=None: semantic_graph_packet(
            graph if ids is None else self._select_graph(ids), self.num_envs if ids is None else len(ids))
        self.relation_runtime = runtime
        self._agent_box_assignment[:] = 0 if self._task_only else 8
        self._group_leaders = torch.zeros(self.num_envs,4,device=self.device,dtype=torch.long)
        self._group_counts = torch.full_like(self._group_leaders,4)
        self._ordinary_keep = torch.arange(0 if self._task_only else 4, device=self.device)[None].expand(self.num_envs, -1).clone()
        self._local_hands = torch.zeros(self.num_envs, 4, 2, 3, device=self.device)
        self._agent_region_assignment = torch.arange(4, device=self.device)[None].expand(self.num_envs, -1).clone()
        self._virtual_door_states = torch.zeros(self.num_envs, 4, 13, device=self.device)
        self._virtual_door_states[..., 6] = 1.

    def _select_graph(self, ids):
        from utils.edge_ontop_spec import select_graph
        return select_graph(self.relation_runtime.graph, ids)

    _amp_family = HumanoidMAOpenPush._amp_family
    _fetch_conditioned_amp_demo = HumanoidMAOpenPush._fetch_conditioned_amp_demo

    def get_open_push_contract(self):
        return dict(revision=2, policy='partition_push_ca4', partition=self._joint_push_spec,
            manifest_sha256=hashlib.sha256(Path(self.cfg['env']['motion_file']).read_bytes()).hexdigest(),
            expert_mapping=self.cfg['env']['ampExpertMapping'], initial_ordinary_boxes=0 if self._task_only else 8, active_ordinary_boxes=0 if self._task_only else 4, task_groups=[1,4], task_tokens=4, bank_size=self._bank_size,
            friction=self.cfg['env']['pushBox']['friction'], plane=self.cfg['env']['plane'])

    def validate_inference_checkpoint(self, checkpoint):
        import copy
        saved = copy.deepcopy(checkpoint.get('open_push_contract'))
        actual = self.get_open_push_contract()
        if self.cfg['args'].test and self.cfg['env'].get('partitionPushAblation', False):
            if not saved or saved.get('policy') != 'partition_push_ca4':
                raise ValueError('Ablation requires a Partition Push checkpoint')
            for key in ('size_min', 'size_max', 'mass_per_agent'):
                saved['partition'][key] = actual['partition'][key]
        if saved != actual:
            raise ValueError('Partition Push checkpoint configuration differs')
        assignment = checkpoint.get('partition_region_assignment')
        if assignment is None:
            print('[Partition Push] Legacy checkpoint: applying exclusive, episode-fixed region assignment', flush=True)
        elif assignment != self.REGION_ASSIGNMENT:
            raise ValueError('Partition Push checkpoint region assignment differs')

    def _reset_ref_state_init(self, slot_env, slot_agent):
        HumanoidMACarry._reset_ref_state_init(self, slot_env, slot_agent)
        root = self._humanoid_root_states[slot_env, slot_agent].clone()
        body = self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent].clone()
        target = self._spawn_roots[slot_env, slot_agent]
        facing = self._spawn_yaw[slot_env, slot_agent]
        turn = facing - yaw_from_quat(root[:, 3:7])
        q = torch.zeros_like(root[:, 3:7]); q[:, 2] = (turn/2).sin(); q[:, 3] = (turn/2).cos()
        target = torch.cat((target, root[:, 2:3]), -1)
        expanded = q[:, None].expand_as(body[..., 3:7])
        body[..., :3] = rotate(expanded, body[..., :3]-root[:, None, :3]) + target[:, None]
        body[..., 3:7] = multiply(expanded, body[..., 3:7])
        body[..., 7:10] = rotate(expanded, body[..., 7:10]); body[..., 10:13] = rotate(expanded, body[..., 10:13])
        self._set_env_state(slot_env, slot_agent, target, multiply(q, root[:, 3:7]),
            self._dof_pos[slot_env, slot_agent].clone(), rotate(q, root[:, 7:10]),
            rotate(q, root[:, 10:13]), self._dof_vel[slot_env, slot_agent].clone())
        self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent] = body

    def _select_regions(self, ids, live=False, reassign=False):
        assignment = self._agent_box_assignment[ids]
        states, _, offsets = group_tokens(self._box_states[ids], self._box_size[ids], assignment, self._ordinary_keep[ids])
        if reassign:
            roots = self._humanoid_root_states[ids] if live else self._kinematic_humanoid_rigid_body_states[ids, :, 0]
            distances = torch.cdist(roots[..., :2], states[:, :4, :2])
            same = assignment[..., None] == assignment[:, None, :]
            self._agent_region_assignment[ids] = exclusive_regions(distances, same)
        strip = self._agent_region_assignment[ids]
        rows = torch.arange(len(ids),device=self.device)[:,None]
        size = self._box_size[ids[:,None],assignment]
        points = self._local_hands[ids].clone(); points.zero_()
        points[..., 0] = offsets[rows,strip,0][...,None]
        points[..., 0, 0] += .15; points[..., 1, 0] -= .15
        points[..., 1] = -size[...,1,None]/2
        self._local_hands[ids] = points
        graph = self.relation_runtime.graph
        token_offset = 8 if getattr(self, '_hidden_door_tokens', False) else 4
        graph.edge_dst[ids, ::2] = strip+token_offset
        graph.edge_src[ids, 1::2] = strip+token_offset

    def _compute_clean_scene_obs(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._select_regions(ids, live=env_ids is None)
        return super()._compute_clean_scene_obs(env_ids)

    def _reset_envs(self, ids):
        if not len(ids): return
        n = len(ids)
        self._reset_default_slots = None
        self._reset_ref_slots = {}; self._reset_ref_motion_ids = {}; self._reset_ref_motion_times = {}
        groups = sample_groups(n, self._joint_push_spec['coupled_probability'], self.device,
                               self._joint_push_spec.get('group_preset','random_scenario'))
        expected, counts, ranks = group_layout(groups, bank_offset=0 if getattr(self, '_task_only', False) else 8)
        self._group_leaders[ids] = groups; self._group_counts[ids] = counts
        boxes = self._initial_box_states[ids].clone(); boxes[..., 7:] = 0
        boxes[..., 2] = self._box_size[ids, :, 2]/2
        # All bank candidates begin parked. Only the four remaining ordinary boxes
        # and one task box per sampled group are placed into the scene.
        boxes[...,0] = 1000.+torch.arange(getattr(self, '_bank_size', 24),device=self.device,dtype=boxes.dtype)*8
        boxes[...,1] = 1000.
        yaw = torch.rand(n,device=self.device)*2*torch.pi
        q = boxes.new_zeros(n,4); q[:,2]=(yaw/2).sin(); q[:,3]=(yaw/2).cos()
        boxes[...,3:7] = q[:,None]
        # Sample eight ordinary positions first, then retain exactly four.
        angle = yaw[:,None]+torch.arange(8,device=self.device)[None]*torch.pi/4
        angle += (torch.rand(n,8,device=self.device)-.5)*.15
        radius = 24+6*torch.rand(n,8,device=self.device)
        ordinary = torch.stack((radius*angle.cos(),radius*angle.sin()),-1)
        order = torch.rand(n,8,device=self.device).argsort(-1)
        keep = order[:,:0 if getattr(self, '_task_only', False) else 4]; self._ordinary_keep[ids] = keep
        rows = torch.arange(n,device=self.device)[:,None]
        boxes[rows,keep,:2] = ordinary.gather(1,keep[...,None].expand(-1,-1,2))
        local_stations = boxes.new_tensor([[-8.,-8.,0.],[8.,-8.,0.],[-8.,8.,0.],[8.,8.,0.]])
        stations = rotate(q[:,None].expand(n,4,4),local_stations[None].expand(n,-1,-1))
        # Group leaders may be any shuffled human; keep their physical bank slots
        # distinct, including two coupled pairs or four solo tasks.
        for agent in range(4):
            active = groups[:,agent] == agent
            r = torch.arange(n,device=self.device)[active]
            boxes[r,expected[active,agent],:2] = stations[active,agent,:2]
        self._box_states[ids] = boxes
        if not hasattr(self,'_spawn_roots'):
            self._spawn_roots = boxes.new_zeros(self.num_envs,4,2)
            self._spawn_yaw = boxes.new_zeros(self.num_envs,4)
        size = self._box_size[ids[:,None],expected]
        local = boxes.new_zeros(n,4,3)
        local[...,0] = size[...,0]*((ranks+.5)/counts-.5)
        lo,hi = self._joint_push_spec['spawn_distance']
        local[...,1] = -size[...,1]/2-lo-(hi-lo)*torch.rand(n,4,device=self.device)
        selected = boxes[rows,expected]
        spawn_mode = self._joint_push_spec.get('spawn_mode', 'rear_lanes')
        if spawn_mode == 'surround_independent':
            positions, headings = sample_surround_spawn(selected, size, (lo, hi),
                self._joint_push_spec.get('spawn_min_agent_distance', 1.2))
            self._spawn_roots[ids] = positions
            self._spawn_yaw[ids] = headings
        elif spawn_mode == 'rear_lanes':
            self._spawn_roots[ids] = (selected[...,:3]+rotate(q[:,None].expand(n,4,4),local))[...,:2]
            self._spawn_yaw[ids] = yaw[:,None]+torch.pi/2
        else:
            raise ValueError('Unknown Partition Push spawn_mode: '+spawn_mode)
        self._reset_actors(ids)
        active = torch.cat((expected,keep),-1)
        candidates = boxes[rows,active]
        nearest = nearest_boxes(self._humanoid_root_states[ids],candidates)
        assignment = active.gather(1,nearest)
        if not torch.equal(assignment,expected):
            raise RuntimeError('Push spawn is not closest to its sampled group box')
        self._agent_box_assignment[ids] = assignment
        lo,hi = self._joint_push_spec['goal_distance']
        local_goal = boxes.new_zeros(n,4,3)
        local_goal[...,1] = lo+(hi-lo)*torch.rand(n,4,device=self.device)
        local_goal = local_goal.gather(1,groups[...,None].expand(-1,-1,3))
        self._tar_pos[ids] = selected[...,:3]+rotate(q[:,None].expand(n,4,4),local_goal)
        self.relation_runtime.graph.edge_dst[ids,1::2] = 4+self.num_objects+groups
        self.relation_runtime.done[ids] = False
        self._select_regions(ids, reassign=True)
        if getattr(self, '_scattered_door_tokens', False):
            self._virtual_door_states[ids] = sample_virtual_doors(
                self._box_states[ids[:, None], self._agent_box_assignment[ids]],
                self._box_size[ids[:, None], self._agent_box_assignment[ids]],
                self._humanoid_root_states[ids])
        self._reset_env_tensors(ids)
        self._compute_observations(ids)
        self._compute_amp_observations(ids)
        self._hist_amp_obs_buf[ids] = self._curr_amp_obs_buf[ids,:,None]
        if getattr(self, 'viewer', None) is not None and (ids == 0).any():
            self._place_partition_camera()

    def render(self, sync_frame_time=False):
        if self.viewer is not None and hasattr(self, '_agent_region_assignment'):
            self._draw_partition_ownership()
        super().render(sync_frame_time)

    def _draw_partition_ownership(self):
        """Paint owned strips with dense surface lines, without adding physics actors."""
        import numpy as np
        self.gym.clear_lines(self.viewer)
        box_colors = np.asarray(self._AGENT_COLOR_PALETTE, dtype=np.float32) * .55
        for eid, env in enumerate(self.envs):
            assignment = self._agent_box_assignment[eid].detach().cpu().tolist()
            regions = self._agent_region_assignment[eid].detach().cpu().tolist()
            states, sizes, _ = group_tokens(self._box_states[eid:eid+1],
                self._box_size[eid:eid+1], self._agent_box_assignment[eid:eid+1],
                self._ordinary_keep[eid:eid+1])
            for bid in set(assignment + self._ordinary_keep[eid].detach().cpu().tolist()):
                owners = [a for a, box in enumerate(assignment) if box == bid]
                color = gymapi.Vec3(*box_colors[owners[0]]) if len(owners) == 1 else gymapi.Vec3(.25, .25, .25)
                self.gym.set_rigid_body_color(env, self._box_handles[eid*self._bank_size+bid],
                                             0, gymapi.MESH_VISUAL, color)
            vertices, colors = [], []
            for agent, region in enumerate(regions):
                # Region states follow the box pose; lift the coating to avoid z fighting.
                half = sizes[0, region].detach().cpu().numpy()/2
                half += .004
                local = []
                for axis in range(3):
                    u, v = [i for i in range(3) if i != axis]
                    for side in (-1, 1):
                        for value in np.linspace(-half[u], half[u], max(2, int(2*half[u]/.008)+1)):
                            a = np.zeros(3); b = np.zeros(3)
                            a[axis] = b[axis] = side*half[axis]
                            a[u] = b[u] = value
                            a[v], b[v] = -half[v], half[v]
                            local.extend((a, b))
                local = torch.as_tensor(np.asarray(local), device=self.device, dtype=states.dtype)
                pose = states[0, region]
                world = rotate(pose[3:7].expand(len(local), -1), local) + pose[:3]
                vertices.append(world.detach().cpu().numpy().reshape(-1, 2, 3))
                colors.append(np.tile(box_colors[agent], (len(local)//2, 1)))
            # Draw above each owned strip so shared-box ownership stays visible.
            roots = self._humanoid_root_states[eid, :, :3].clone()
            roots[:, 2] += .2
            region_ids = self._agent_region_assignment[eid]
            endpoints = states[0, region_ids, :3].clone()
            top = endpoints.new_zeros(self.num_agents, 3)
            top[:, 2] = sizes[0, region_ids, 2]/2 + .06
            endpoints += rotate(states[0, region_ids, 3:7], top)
            links = torch.stack((roots, endpoints), 1)
            vertices.append(links.detach().cpu().numpy())
            colors.append(np.asarray(self._AGENT_COLOR_PALETTE[:self.num_agents], dtype=np.float32)*.75)
            vertices = np.concatenate(vertices).astype(np.float32)
            colors = np.concatenate(colors).astype(np.float32)
            self.gym.add_lines(self.viewer, env, len(vertices), vertices, colors)

    def _init_camera(self):
        # The useful scene positions are assigned at reset, after construction.
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0, :3].detach().cpu().numpy().copy()
        self.cfg['env']['cameraFollow'] = False
        self._camera_follow_flag = False

    def _place_partition_camera(self):
        roots = self._humanoid_root_states[0, :, :3]
        boxes = self._box_states[0, self._agent_box_assignment[0], :3]
        # Ignore parked bank actors at x/y >= 1000 and distant distractors.
        points = torch.cat((roots, boxes, self._tar_pos[0]), 0)
        lower, upper = points.amin(0), points.amax(0)
        target = (lower + upper) / 2
        target[2] = 1.
        distance = max(8., float((upper[:2] - lower[:2]).norm()) * 1.2)
        camera = target + target.new_tensor([0., -distance, distance * .75])
        self.gym.viewer_camera_look_at(self.viewer, None,
            gymapi.Vec3(*camera.detach().cpu().tolist()),
            gymapi.Vec3(*target.detach().cpu().tolist()))
        self._camera_follow_flag = False
        self._cam_prev_char_pos = roots[0].detach().cpu().numpy().copy()
        print('[Partition Push viewer] camera framed active groups', flush=True)

    def _compute_reward(self, actions):
        self._select_regions(torch.arange(self.num_envs, device=self.device), live=True)
        spec = self._joint_push_spec['reward']
        r = partition_reward(self._humanoid_root_states, self._rigid_body_pos[:, :, self._key_body_ids[:2]],
            self._box_states, self._tar_pos, self._box_size, self._agent_box_assignment, self._local_hands, spec)
        runtime = self.relation_runtime
        runtime.phi.copy_(r['phi'].flatten(1)); runtime.own_success.copy_(r['success'].flatten(1))
        runtime.achieved.copy_(runtime.own_success); runtime.done.copy_(r['done'])
        power = -self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(r['reward'])
        collision = -self._agent_collision_coeff*cpa_penalty(self._humanoid_root_states[..., :3],
            self._humanoid_root_states[..., 7:10], self._agent_collision_dist, .99, self.dt) if self._agent_collision_penalty else torch.zeros_like(power)
        total = r['reward']+power+collision
        self.rew_buf[:] = total.flatten()
        terms = torch.stack((r['state'].sum(-1)*spec['state_reward_weight'],
            r['progress'].sum(-1)*spec['progress_reward_weight'],
            r['success'].float().sum(-1)*spec['success_reward_weight'], power, collision, total), -1).reshape(-1, 6)
        self.extras['reward_terms'] = terms
        self._reward_term_sums += terms.sum(0); self._reward_term_count += len(terms)
        self.extras['partition_push_group_counts'] = self._group_counts.clone()
        self.extras['partition_push_assignment'] = self._agent_box_assignment.clone()
        self.extras['partition_push_regions'] = self._agent_region_assignment.clone()
        self.extras['partition_push_hand_targets'] = r['targets']
        self.extras['partition_push_group_phi'] = r['group_phi']

    def consume_relation_diagnostics(self): return {}

    def _finish_relation_diagnostics(self): pass

    def _compute_metrics_evaluation(self):
        self._success_buf.copy_(self.relation_runtime.done.flatten().long())
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        self._precision_buf.copy_((self._box_states[rows, self._agent_box_assignment, :2]-self._tar_pos[..., :2]).norm(dim=-1).flatten())
