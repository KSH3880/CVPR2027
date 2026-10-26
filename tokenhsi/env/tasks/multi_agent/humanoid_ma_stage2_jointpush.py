import torch
from isaacgym import gymapi

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi
from env.tasks.multi_agent.humanoid_ma_stage1_jointpush import HumanoidMAStage1JointPush
from env.tasks.multi_agent.stage2_jointpush_reward import Stage2JointPushRuntime
from utils import stage2_jointpush as spec
from utils import stage2_jointpush_scene as scene
from utils.edge_ontop_spec import copy_graph_rows
from utils.jointpush_geometry import push_rear_xy, push_face_geometry, rotate


class HumanoidMAStage2JointPush(HumanoidMACarry):
    _solo_cache_class = LocoRsi
    _motion_skills_to_load = HumanoidMAStage1JointPush._motion_skills_to_load
    _record_reference = HumanoidMAStage1JointPush._record_reference
    _amp_family = HumanoidMAStage1JointPush._amp_family

    def __init__(self, cfg, *args, **kwargs):
        self._amp_num_families = 4
        self._joint = cfg['env']['jointPush']
        super().__init__(cfg, *args, **kwargs)

    def get_relation_suffix_size(self):
        return 6*self.num_agents

    def _load_box_asset(self):
        self._box_spawn_radius = 4.5
        self._box_size, masses = spec.asset_bank(self.num_envs, self.cfg['env'], self.device)
        self._box_scale = self._box_size/self._box_size.new_tensor(self._build_base_size)
        self._box_density = (masses/self._box_size.prod(-1)).flatten()
        self._box_assets = []
        cache = {}
        for size, rho in zip(self._box_size.flatten(0,1).cpu().tolist(),self._box_density.cpu().tolist()):
            key = tuple(size)+(rho,)
            if key not in cache:
                options = gymapi.AssetOptions()
                options.density = rho
                options.angular_damping = options.linear_damping = .01
                options.max_angular_velocity = 100.
                options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                cache[key] = self.gym.create_box(self.sim, *size, options)
            self._box_assets.append(cache[key])

    def _build_box(self, env_id, object_id, env_ptr):
        super()._build_box(env_id, object_id, env_ptr)
        handle = self._box_handles[-1]
        if object_id >= 16:
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            for prop in props:
                prop.friction = self._joint['box']['friction']
                prop.restitution = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, props)
        state = self.gym.get_actor_rigid_body_states(env_ptr, handle, gymapi.STATE_ALL)
        state['pose']['p']['x'][:] = 1000.+12*object_id
        state['pose']['p']['y'][:] = 1000.
        self.gym.set_actor_rigid_body_states(env_ptr, handle, state, gymapi.STATE_ALL)

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        self._groups = self._agent_box_assignment.clone()
        self._members = torch.ones_like(self._groups)
        self._regions = torch.zeros_like(self._groups)
        self._templates_by_owner = torch.full_like(self._groups,3)
        self._logical_box_order[:,4:] = torch.arange(12,16,device=self.device)

    def _logical_box_values(self, values, env_ids=None):
        ids = torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        result = super()._logical_box_values(values, ids).clone()
        if values is self._box_states:
            sizes = self._assigned_box_values(self._box_size,ids)
            result[:,:4] = spec.region_states(result[:,:4],sizes,self._members[ids],self._regions[ids])[0]
        elif values is self._box_size:
            result[:,:4,0] /= self._members[ids]
        elif values is self._box_bps:
            signs = result.new_tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],
                                      [1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]])
            result = self._logical_box_values(self._box_size,ids)[...,None,:]*signs/2
        return result

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self.relation_runtime = Stage2JointPushRuntime(self.num_envs,self.relation_runtime.graph,
            self._relation_cfg,self.device,contact_weights=self._joint['contact_weights'])
        self._push_hits = torch.zeros(self.num_envs,4,device=self.device,dtype=torch.bool)

    def _sample_episode_graph(self, ids):
        preset = self._task_graph_preset
        groups = spec.sample_groups(len(ids),4,.8,self.device,
            preset if preset in ('random_scenario','all_coupled','all_solo') else 'all_solo')
        weights = torch.tensor([0.,.1,.2,.25,.2,.25],device=self.device)
        templates = spec.sample_templates(groups,weights)
        forced = dict(sit=1,climb=2,holding_at=3,holding_ontop=4,push=5)
        if preset in forced:
            templates.fill_(forced[preset])
        self._groups[ids],self._members[ids] = groups,spec.group_sizes(groups)
        self._regions[ids] = spec.assign_regions(groups)
        self._templates_by_owner[ids] = templates
        assignment = spec.physical_assignment(templates,groups)
        self._agent_box_assignment[ids] = assignment
        self._logical_box_order[ids,:4] = assignment
        graph = spec.compose_graph(templates,groups,self._relation_graph_spec['shuffle_edge_order'])
        copy_graph_rows(self.relation_runtime.graph,ids,graph)
        self._edge_goal_owners[ids] = True
        self._ontop_maintain_steps[ids] = 0
        self._sampling_counts[0] += len(ids)
        self._sampling_counts[1:7] += torch.bincount(templates.flatten(),minlength=6)
        self._push_hits[ids] = False
        self._update_box_assignment_colors(ids)

    def _reset_ref_state_init(self, env, owner):
        push = self._templates_by_owner[env,owner] == 5
        for mask, name in ((~push,'loco'),(push,'push_loco')):
            es, humans = env[mask],owner[mask]
            if not len(es):
                continue
            lib = self._motion_lib[name]
            mids = lib.sample_motions(len(es))
            times = lib.sample_time_rsi(mids)
            root,q,dof,vel,ang,dof_vel,_ = lib.get_motion_state(mids,times)
            body,bq,bv,ba = lib.get_motion_state_max(mids,times)
            offset = self._agent_spawn_offsets[humans]+self._env_origins[es]
            self._set_env_state(es,humans,root+offset,q,dof,vel,ang,dof_vel)
            self._kinematic_humanoid_rigid_body_states[es,humans] = torch.cat((body+offset[:,None],bq,bv,ba),-1)
            self._every_env_init_dof_pos[es,humans] = dof
            self._record_reference('loco',es,humans,mids,times)

    def _reset_all_boxes_random_arena(self, ids):
        self._box_states[ids] = 0.
        self._box_states[ids,:,6] = 1.
        self._box_states[ids,:,2] = self._box_size[ids,:,2]/2
        self._box_states[ids,:,:2] = self._env_origins[ids,None,:2]+1000.
        self._box_states[ids,:,0] += torch.arange(24,device=self.device)*12
        for slot in range(12,16):
            self._box_states[ids,slot,:2] = self._sample_arena_xy(self._env_origins[ids,:2])
            yaw = torch.rand(len(ids),device=self.device)*2*torch.pi
            self._box_states[ids,slot,5] = (yaw/2).sin()
            self._box_states[ids,slot,6] = (yaw/2).cos()

    def _reset_ontop_context_envs(self, ids):
        if not len(ids):
            return
        self._sample_episode_graph(ids)
        self._reset_default_slots = None
        self._reset_ref_slots,self._reset_ref_motion_ids,self._reset_ref_motion_times = {},{},{}
        self._reset_actors(ids)
        saved = (self._reset_ref_slots.copy(),self._reset_ref_motion_ids.copy(),self._reset_ref_motion_times.copy())
        pending = ids
        for _ in range(self.cfg['env']['stage2JointPush']['layout_attempts']):
            self._reset_ref_slots,self._reset_ref_motion_ids,self._reset_ref_motion_times = (v.copy() for v in saved)
            self._keep_reset_slots_for_envs(pending)
            self._reset_boxes(pending)
            self._reset_task(pending)
            self._configure_ontop_targets(pending)
            rejected = scene.sample_push(self,pending)
            scene.place_distractors(self,pending)
            from utils.stage1_platforms import reset_stage1_platforms
            reset_stage1_platforms(self,pending)
            rejected |= scene.scene_infeasible(self,pending)
            self._sampling_retries += rejected.sum()
            pending = pending[rejected]
            if not len(pending):
                break
        if len(pending):
            raise RuntimeError('Stage2 jointpush layout exhausted; people/group retained: '+str(pending[:16].tolist()))
        self._reset_ref_slots,self._reset_ref_motion_ids,self._reset_ref_motion_times = saved
        if not self._is_eval:
            templates = self._templates_by_owner[ids].flatten()
            self._size_rsi_executed[:,0] += torch.bincount(templates,minlength=6)
        self._reset_env_tensors(ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(ids)
        self._compute_observations(ids)
        self._init_amp_obs(ids)

    def _ontop_scene_infeasible(self, ids):
        return scene.scene_infeasible(self,ids)

    def _contact_geometry(self, ids, bodies):
        hands = bodies[:,:,self._key_body_ids[:2]]
        boxes = self._assigned_box_values(self._box_states,ids)
        sizes = self._assigned_box_values(self._box_size,ids)
        tokens, cells = spec.region_states(boxes,sizes,self._members[ids],self._regions[ids])
        local = spec.inverse_rotate(tokens[...,None,3:7].expand(*hands.shape[:-1],4),hands-tokens[...,None,:3])
        dx = (local[...,0].abs()-cells[...,None,0]/2).clamp_min(0)
        distance = (dx.square()+(local[...,1]+cells[...,None,1]/2).square()).sqrt().amax(-1)
        normal = torch.zeros_like(sizes)
        normal[...,1] = 1.
        normal = rotate(boxes[...,3:7],normal)[...,:2]
        shared = self._members[ids] > 1
        distance = torch.where(shared,distance,push_rear_xy(hands,boxes,self._tar_pos[ids],sizes).amax(-1))
        normal = torch.where(shared[...,None],normal,push_face_geometry(hands,boxes,self._tar_pos[ids],sizes)[1])
        return distance,normal,hands,boxes,sizes

    def _evaluate_ontop_context(self, env_ids=None):
        phi,diag = super()._evaluate_ontop_context(env_ids)
        ids = torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        bodies = self._rigid_body_pos[ids] if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids,...,:3]
        distance,_,hands,boxes,sizes = self._contact_geometry(ids,bodies)
        self.relation_runtime.gate[ids] = spec.group_contact_gate(hands,boxes,sizes,self._tar_pos[ids],self._groups[ids])
        owner = self.relation_runtime.graph.edge_owner[ids]
        relation = self.relation_runtime.graph.edge_relation[ids]
        root_distance = (bodies[:,:,0,:2]-self._logical_box_values(self._box_states,ids)[:,:4,:2]).norm(dim=-1)
        contact = self._joint['contact']
        push_at = (relation == 7) & (self._templates_by_owner[ids] == 5).gather(1,owner)
        delta = self._tar_pos[ids]-boxes[...,:3]
        for mask, state, progress, dist, xy, z in (
            (relation == spec.PUSH,torch.exp(-contact['near_distance_scale']*distance.square()),
             1/(1+(root_distance-contact['progress_delta']).clamp_min(0)/contact['progress_sigma']),distance,root_distance,distance*0),
            (push_at,torch.exp(-10*delta.square().sum(-1)),1/(1+(delta[...,:2].norm(dim=-1)-.1).clamp_min(0)),
             delta.norm(dim=-1),delta[...,:2].norm(dim=-1),delta[...,2].abs())):
            phi = torch.where(mask,state.gather(1,owner),phi)
            for key,value in (('progress',progress),('distance',dist),('distance_xy',xy),('z_error',z)):
                diag[key] = torch.where(mask,value.gather(1,owner),diag[key])
        return phi,diag

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._push_hits.zero_()

    def _physics_step(self):
        ids = torch.arange(self.num_envs,device=self.device)
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim,True)
            self._refresh_sim_tensors()
            distance,normal,_,_,_ = self._contact_geometry(ids,self._rigid_body_pos)
            reaction = -(self._contact_forces[:,:,self._key_body_ids[:2],:2]*normal[:,:,None]).sum(-1)
            force = (self._assigned_box_values(self._ontop_box_contact)[...,:2]*normal).sum(-1)
            cfg = self._joint['contact']
            hit = (torch.exp(-cfg['near_distance_scale']*distance.square()) > cfg['satisfaction_threshold'])
            self._push_hits |= hit & (reaction >= cfg['min_hand_force']).all(-1) & (force >= cfg['min_box_force'])

    def _compute_reward(self, actions):
        self.relation_runtime.contact.copy_(self._push_hits)
        return super()._compute_reward(actions)

    def _compute_clean_scene_obs(self, env_ids=None):
        raw = super()._compute_clean_scene_obs(env_ids)
        groups = self._groups if env_ids is None else self._groups[env_ids]
        return torch.cat((raw,groups.float()),-1)

    def _init_amp_obs_ref(self, slots, motion_ids, motion_times, skill_name):
        push = self._templates_by_owner[slots] == 5
        for selected,name in ((~push,'loco'),(push,'push_loco')):
            if selected.any():
                super()._init_amp_obs_ref(tuple(v[selected] for v in slots),motion_ids[selected],motion_times[selected],name)

    def _fetch_conditioned_amp_demo(self, count):
        from utils.unified_training import append_family
        # Expected per-agent prior of the sequential 0.8 sampler with four people.
        solo = .17733333333333334
        weights = torch.tensor([solo*.45,solo*.1,solo*.2,1-solo+solo*.25],device=self.device)
        family = torch.multinomial(weights,count,replacement=True)
        result = torch.empty(count,self.get_num_amp_obs(),device=self.device)
        for index,name in enumerate(('carry','sit','climb','push')):
            rows = (family == index).nonzero().flatten()
            if not len(rows):
                continue
            probs = torch.tensor(self.cfg['env']['ampExpertMapping'][name],device=self.device)
            selected = torch.multinomial(probs,len(rows),replacement=True)
            for skill,libname in enumerate(self.cfg['env']['skill']):
                dest = rows[selected == skill]
                if not len(dest):
                    continue
                lib = self._motion_lib['push_loco' if index == 3 and libname == 'loco' else libname]
                mids = lib.sample_motions(len(dest))
                history = self.dt*(self._num_amp_obs_steps-1)
                times = lib.sample_time(mids,truncate_time=history)+history
                obs = self.build_amp_obs_demo(mids,times,lib).reshape(len(dest),self._num_amp_obs_steps,-1)
                result[dest] = append_family(obs,family[dest,None].expand(-1,self._num_amp_obs_steps),4).flatten(1)
        return result

    def _video_focus_points(self):
        return torch.cat((self._humanoid_root_states[0,:,:3],self._logical_box_values(self._box_states,torch.tensor([0],device=self.device))[0,:,:3]),0)
