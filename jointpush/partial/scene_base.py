from types import SimpleNamespace
from isaacgym import gymapi, gymtorch
import torch
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from jointpush.partial.transforms import rotate, multiply, yaw_from_quat
from jointpush.partial.geometry import *

class PartitionSceneBase(HumanoidMACarry):
    REGION_ASSIGNMENT = 'exclusive_reset_random_v1'

    REWARD_TERM_NAMES = ('state', 'progress', 'success', 'power', 'collision', 'total')

    def _load_box_asset(self):
            spec = self._joint_push_spec
            self._box_size, self._payload_masses = sample_asset_bank(self.num_envs, spec, self.device)
            self._box_assets = []
            for size, mass in zip(self._box_size.flatten(0, 1).cpu().tolist(), self._payload_masses.flatten().cpu().tolist()):
                options = gymapi.AssetOptions()
                options.density = mass / (size[0]*size[1]*size[2])
                options.angular_damping = options.linear_damping = .01
                self._box_assets.append(self.gym.create_box(self.sim, *size, options))

    def _build_env(self, env_id, env_ptr, humanoid_asset):
            HumanoidMA._build_env(self, env_id, env_ptr, humanoid_asset)
            for box_id in range(24):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0., 0.,
                                     float(self._box_size[env_id, box_id, 2])/2)
                handle = self.gym.create_actor(env_ptr, self._box_assets[env_id*24+box_id], pose,
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
            self._box_states = self._root_states.view(self.num_envs, actors, 13)[:, 4:28]
            self._box_actor_ids = (actors*torch.arange(self.num_envs, device=self.device, dtype=torch.int32)[:, None]
                                   + torch.arange(24, device=self.device, dtype=torch.int32)[None] + 4)
            self._initial_box_states = self._box_states.clone()
            self._build_box_bps()

    def _build_marker_state_tensors(self):
            actors = self.get_num_actors_per_env()
            self._marker_states = self._root_states.view(self.num_envs, actors, 13)[:, 28:32]
            self._marker_pos = self._marker_states[..., :3]
            self._marker_actor_ids = (actors*torch.arange(self.num_envs, device=self.device, dtype=torch.int32)[:, None]
                                      + torch.arange(28, 32, device=self.device, dtype=torch.int32)[None])

    def _logical_box_values(self, values, env_ids=None):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            states, sizes, _ = group_tokens(self._box_states[ids], self._box_size[ids],
                self._agent_box_assignment[ids], self._ordinary_keep[ids])
            if values is self._box_states: return states
            if values is self._box_size: return sizes
            if values is self._box_bps:
                signs = self._box_bps.new_tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],
                                                 [1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]])
                return sizes[..., None, :]*signs/2
            raise ValueError('Unknown partitioned object field')

    def _select_graph(self, ids):
            from utils.edge_ontop_spec import select_graph
            return select_graph(self.relation_runtime.graph, ids)

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
                same = assignment[..., None] == assignment[:, None, :]
                self._agent_region_assignment[ids] = random_regions(same)
            strip = self._agent_region_assignment[ids]
            rows = torch.arange(len(ids),device=self.device)[:,None]
            size = self._box_size[ids[:,None],assignment]
            points = self._local_hands[ids].clone(); points.zero_()
            points[..., 0] = offsets[rows,strip,0][...,None]
            points[..., 0, 0] += .15; points[..., 1, 0] -= .15
            points[..., 1] = -size[...,1,None]/2
            self._local_hands[ids] = points
            graph = self.relation_runtime.graph
            graph.edge_dst[ids, ::2] = strip+4
            graph.edge_src[ids, 1::2] = strip+4

    def _reset_envs(self, ids):
            if not len(ids): return
            n = len(ids)
            self._reset_default_slots = None
            self._reset_ref_slots = {}; self._reset_ref_motion_ids = {}; self._reset_ref_motion_times = {}
            groups = sample_groups(n, self._joint_push_spec['coupled_probability'], self.device,
                                   self._joint_push_spec.get('group_preset','random_scenario'))
            expected, counts, ranks = group_layout(groups)
            self._group_leaders[ids] = groups; self._group_counts[ids] = counts
            boxes = self._initial_box_states[ids].clone(); boxes[..., 7:] = 0
            boxes[..., 2] = self._box_size[ids, :, 2]/2
            # All bank candidates begin parked. Only the four remaining ordinary boxes
            # and one task box per sampled group are placed into the scene.
            boxes[...,0] = 1000.+torch.arange(24,device=self.device,dtype=boxes.dtype)*8
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
            keep = order[:,:4]; self._ordinary_keep[ids] = keep
            rows = torch.arange(n,device=self.device)[:,None]
            boxes[rows,keep,:2] = ordinary.gather(1,keep[...,None].expand(-1,-1,2))
            local_stations = boxes.new_tensor([[-8.,-8.,0.],[8.,-8.,0.],[-8.,8.,0.],[8.,8.,0.]])
            stations = rotate(q[:,None].expand(n,4,4),local_stations[None].expand(n,-1,-1))
            station_order = torch.rand(n,4,device=self.device).argsort(-1)
            stations = stations.gather(1,station_order[...,None].expand(-1,-1,3))
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
            self._agent_box_assignment[ids] = expected
            size = self._box_size[ids[:,None],expected]
            selected = boxes[rows,expected]
            lo,hi = self._joint_push_spec['goal_distance']
            local_goal = boxes.new_zeros(n,4,3)
            local_goal[...,1] = lo+(hi-lo)*torch.rand(n,4,device=self.device)
            local_goal = local_goal.gather(1,groups[...,None].expand(-1,-1,3))
            self._tar_pos[ids] = selected[...,:3]+rotate(q[:,None].expand(n,4,4),local_goal)
            if self._joint_push_spec['spawn_mode'] != 'goal_opposite_180':
                raise ValueError('Partial requires goal-opposite 180-degree spawns')
            positions,headings = sample_rear_spawn(selected,size,self._tar_pos[ids],
                self._joint_push_spec['spawn_distance'],self._joint_push_spec['spawn_min_agent_distance'],
                self._joint_push_spec['spawn_clearance'])
            self._spawn_roots[ids] = positions
            self._spawn_yaw[ids] = headings
            self._reset_actors(ids)
            self.relation_runtime.graph.edge_dst[ids,1::2] = 12+groups
            self.relation_runtime.done[ids] = False
            self._select_regions(ids, reassign=True)
            self._reset_env_tensors(ids)
            self._compute_observations(ids)
            self._compute_amp_observations(ids)
            self._hist_amp_obs_buf[ids] = self._curr_amp_obs_buf[ids,:,None]
            if getattr(self, 'viewer', None) is not None and (ids == 0).any():
                self._place_partition_camera()

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

    def consume_relation_diagnostics(self): return {}

    def _finish_relation_diagnostics(self): pass

    def _compute_metrics_evaluation(self):
            self._success_buf.copy_(self.relation_runtime.done.flatten().long())
            rows = torch.arange(self.num_envs, device=self.device)[:, None]
            self._precision_buf.copy_((self._box_states[rows, self._agent_box_assignment, :2]-self._tar_pos[..., :2]).norm(dim=-1).flatten())
    def __init__(self,cfg,*args,**kwargs):
        self._joint_push_spec=cfg['env']['partitionPush']
        super().__init__(cfg,*args,**kwargs)
        self._stage2=self._shared_support=True
        self.REWARD_TERM_NAMES=('state','progress','success','power','collision','total')
        self._reward_term_sums=torch.zeros(6,device=self.device)
        self._reward_term_count=0
        self._env_origins.zero_()

    def _create_envs(self,*args,**kwargs):
        self.num_physical_objects=24
        self._reset_random_height=self._stage1_platforms=self._fixed_platforms=False
        return super()._create_envs(*args,**kwargs)

    def _build_platform_state_tensors(self): pass
    def _draw_ontop_context(self):
        if self.viewer is None or not hasattr(self, '_agent_region_assignment'):
            return
        import numpy as np
        self.gym.clear_lines(self.viewer)
        tokens, sizes, _ = group_tokens(self._box_states, self._box_size,
                                       self._agent_box_assignment, self._ordinary_keep)
        for e, env in enumerate(self.envs):
            lines, colors = [], []
            for a in range(self.num_agents):
                region = int(self._agent_region_assignment[e, a])
                state, size = tokens[e, region], sizes[e, region]
                # Mark the allocated strip on the top face of its physical box.
                local = state.new_tensor([[-.5,-.5, .5],[.5,-.5,.5],
                                          [.5,.5,.5],[-.5,.5,.5]]) * size
                local[:, 2] += .035
                corners = rotate(state[3:7].expand(4,-1), local) + state[:3]
                target = corners.mean(0)
                root = self._humanoid_root_states[e,a,:3].clone()
                root[2] += .3
                color = self._AGENT_COLOR_PALETTE[a]
                for j in range(4):
                    lines.append(torch.stack((corners[j], corners[(j+1)%4])))
                    colors.append(color)
                lines.append(torch.stack((root,target)))
                colors.append(color)
            vertices = torch.stack(lines).detach().cpu().numpy().astype(np.float32)
            self.gym.add_lines(self.viewer, env, len(lines), vertices,
                               np.asarray(colors, dtype=np.float32))

    def pre_physics_step(self,actions): return HumanoidMA.pre_physics_step(self,actions)
    def get_clean_object_obs_size(self): return 30
    def _clean_object_features(self,object_local,env_ids): return object_local
    def get_relation_suffix_size(self): return 24

    def _init_relation_runtime(self):
        from utils.edge_context_spec import EdgeContextGraph,PUSH_CONTACT,AT
        m,o=4,8
        src=torch.tensor([v for a in range(m) for v in (a,m+a)],device=self.device)
        dst=torch.tensor([v for a in range(m) for v in (m+a,m+o+a)],device=self.device)
        rel=torch.tensor([PUSH_CONTACT,AT]*m,device=self.device)
        owner=torch.arange(m,device=self.device).repeat_interleave(2)
        valid=torch.ones(8,device=self.device,dtype=torch.bool)
        graph=EdgeContextGraph(tuple('edge'+str(i) for i in range(8)),m,o,src,dst,rel,owner,valid,rel==AT,
            torch.zeros(8,8,device=self.device,dtype=torch.bool),torch.full_like(src,-1))
        from utils.edge_ontop_spec import expand_graph
        graph=expand_graph(graph,self.num_envs)
        runtime=SimpleNamespace(graph=graph)
        for key in ('phi','error'):setattr(runtime,key,torch.zeros(self.num_envs,8,device=self.device))
        for key in ('own_success','achieved'):setattr(runtime,key,torch.zeros(self.num_envs,8,device=self.device,dtype=torch.bool))
        runtime.done=torch.zeros(self.num_envs,4,device=self.device,dtype=torch.bool)
        def suffix(ids=None):
            ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
            packet=torch.ones(len(ids),4,5,device=self.device)
            packet[...,1]=4
            packet[...,2]=torch.arange(4,device=self.device)
            packet[...,3]=graph.edge_dst[ids,::2]
            packet[...,4]=graph.edge_dst[ids,1::2]
            return packet.flatten(1)
        runtime.suffix=suffix
        self.relation_runtime=runtime
        self._agent_box_assignment[:]=8
        self._group_leaders=torch.zeros(self.num_envs,4,device=self.device,dtype=torch.long)
        self._group_counts=torch.full_like(self._group_leaders,4)
        self._ordinary_keep=torch.arange(4,device=self.device)[None].expand(self.num_envs,-1).clone()
        self._local_hands=torch.zeros(self.num_envs,4,2,3,device=self.device)
        self._agent_region_assignment=torch.arange(4,device=self.device)[None].expand(self.num_envs,-1).clone()

    def _compute_clean_scene_obs(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        self._select_regions(ids,live=env_ids is None)
        scene=HumanoidMACarry._compute_clean_scene_obs(self,env_ids)
        return torch.cat((scene,self._agent_box_assignment[ids].float()),-1)

    def _reset_env_tensors(self,ids):
        HumanoidMA._reset_env_tensors(self,ids)
        actors=[self._box_actor_ids[ids].contiguous().view(-1)]
        if self._enable_markers:
            self._marker_pos[ids]=self._tar_pos[ids]
            actors.append(self._marker_actor_ids[ids].contiguous().view(-1))
        actors=torch.cat(actors)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self._root_states),gymtorch.unwrap_tensor(actors),len(actors))
        if self._is_eval:
            self._success_buf[self._flat_slot_ids(ids)]=0
            self._precision_buf[self._flat_slot_ids(ids)]=float('inf')

    def _amp_family(self,env_ids=None,slot_agent=None):
        if env_ids is None:count=self.num_envs*self.num_agents
        else:count=len(env_ids)*(self.num_agents if slot_agent is None else 1)
        return torch.full((count,),3,device=self.device,dtype=torch.long)
