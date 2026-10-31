"""Original BEFORE coordination with Stage1 PUSH only on independent people."""
import torch
from isaacgym import gymapi
from env.tasks.multi_agent.humanoid_ma_before import HumanoidMABefore
from env.tasks.multi_agent.humanoid_ma_push import HumanoidMAPush
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from env.tasks.multi_agent.jointpush_reward import contact_state
from utils.before_push_spec import sample_graph
from utils.before_push_layout import sample_layout, paths_valid, PushLayoutError
from utils.jointpush_scene import goal_angles_valid
from utils.jointpush_geometry import push_rear_xy


class HumanoidMABeforePush(HumanoidMABefore):
    def __init__(self,cfg,*args,**kwargs):
        self._amp_num_families=4
        self._joint=cfg['env']['jointPush']
        super().__init__(cfg,*args,**kwargs)

    def get_relation_suffix_size(self):
        return 6*self.num_agents

    def _compute_clean_scene_obs(self,env_ids=None):
        scene=super()._compute_clean_scene_obs(env_ids)
        pre=self._before_predecessor if env_ids is None else self._before_predecessor[env_ids]
        groups=torch.arange(self.num_agents,device=self.device)[None].expand_as(pre)
        groups=torch.where(pre>=0,pre,groups)
        return torch.cat((scene,groups.float()),-1)

    def _load_box_asset(self):
        n,m,p=self.num_envs,self.num_agents,self.num_physical_objects
        self._excluded_distractor_range=(4*m,5*m)
        self._box_spawn_radius=4.5
        self._box_size=torch.empty(n,p,3,device=self.device)
        ranges=self.cfg['env']['box']['build']['taskSizeRanges']
        for bank,name in enumerate(('HOLDING_AT','SIT','CLIMB','SUPPORT')):
            bounds=[list(v) for v in ranges[name]]
            family=self._task_graph_preset.rsplit('_',1)[-1].upper()
            if bank==0 and family in ('SIT','CLIMB') and self._task_graph_preset.startswith(('before_','one_pair_')):
                bounds=[[max(lo,a),min(hi,b)] for (lo,hi),(a,b) in zip(bounds,ranges[family])]
            for axis,(lo,hi) in enumerate(bounds):
                self._box_size[:,bank*m:(bank+1)*m,axis]=torch.randint(round(lo*20),round(hi*20)+1,(n,m),device=self.device)/20
        profiles=torch.tensor(self._joint['box']['profiles'],device=self.device)
        self._box_size[:,4*m:5*m]=profiles[torch.randint(len(profiles),(n,m),device=self.device)]
        for axis,(lo,hi) in enumerate(ranges['SUPPORT']):
            self._box_size[:,5*m:,axis]=torch.randint(round(lo*20),round(hi*20)+1,(n,p-5*m),device=self.device)/20
        density=torch.full((n,p),100.,device=self.device);density[:,4*m:5*m]=self._joint['box']['density']
        self._box_density=density.flatten()
        self._box_scale=self._box_size/self._box_size.new_tensor(self._build_base_size)
        self._box_assets=[];cache={}
        for dims,rho in zip(self._box_size.flatten(0,1).cpu().tolist(),self._box_density.cpu().tolist()):
            key=tuple(dims)+(rho,)
            if key not in cache:
                options=gymapi.AssetOptions();options.density=rho
                options.angular_damping=options.linear_damping=.01
                options.max_angular_velocity=100.
                options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                cache[key]=self.gym.create_box(self.sim,*dims,options)
            self._box_assets.append(cache[key])

    def _build_box(self,env_id,object_id,env_ptr):
        super()._build_box(env_id,object_id,env_ptr)
        if 4*self.num_agents<=object_id<5*self.num_agents:
            props=self.gym.get_actor_rigid_shape_properties(env_ptr,self._box_handles[-1])
            for prop in props:prop.friction=self._joint['box']['friction'];prop.restitution=0.
            self.gym.set_actor_rigid_shape_properties(env_ptr,self._box_handles[-1],props)

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        self._push_goals=torch.zeros(self.num_envs,self.num_agents,3,device=self.device)
        self._push_hits=torch.zeros(self.num_envs,self.num_agents,dtype=torch.bool,device=self.device)
        self._push_layout_failed=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._solo_counts=torch.zeros(6,device=self.device)

    def _draw_episode_graph(self,ids):
        graph,pre,templates=sample_graph(self._box_size[ids,:self.num_agents],
            self.cfg['env']['box']['build']['taskSizeRanges'],self._relation_graph_spec['template_probabilities'],self._task_graph_preset,self.num_objects,
            pairing_probability=self.cfg['env']['before']['pairing_probability'])
        paired=pre>=0
        rows,followers=paired.nonzero(as_tuple=True)
        paired[rows,pre[rows,followers]]=True
        self._solo_counts+=torch.bincount(templates[~paired],minlength=6)
        return graph,pre,templates

    def _primary_objects(self,templates):
        result=super()._primary_objects(templates)
        people=torch.arange(self.num_agents,device=self.device)[None]
        return torch.where(templates==5,4*self.num_agents+people,result)

    def _init_ontop_context_runtime(self):
        super()._init_ontop_context_runtime()
        self.relation_runtime=BeforePushRuntime(self.num_envs,self.relation_runtime.graph,self._relation_cfg,self.device,contact_weights=self._joint["contact_weights"])

    def _reset_ref_state_init(self,env,owner):
        push=self._templates[env,owner]==5
        if (~push).any():super()._reset_ref_state_init(env[~push],owner[~push])
        e,a=env[push],owner[push]
        if not len(e):return
        lib=self._motion_lib['push_loco'];mids=lib.sample_motions(len(e));times=lib.sample_time_rsi(mids)
        root,q,dof,vel,ang,dof_vel,_=lib.get_motion_state(mids,times)
        body,bq,bv,ba=lib.get_motion_state_max(mids,times)
        offset=self._agent_spawn_offsets[a]+self._env_origins[e]
        self._set_env_state(e,a,root+offset,q,dof,vel,ang,dof_vel)
        self._kinematic_humanoid_rigid_body_states[e,a]=torch.cat((body+offset[:,None],bq,bv,ba),-1)
        self._every_env_init_dof_pos[e,a]=dof
        HumanoidMAPush._record_reference(self,'loco',e,a,mids,times)

    def _init_amp_obs_ref(self,slots,mids,times,skill):
        if skill!='loco':return super()._init_amp_obs_ref(slots,mids,times,skill)
        push=self._templates[slots]==5
        for chosen,name in ((~push,'loco'),(push,'push_loco')):
            if chosen.any():super()._init_amp_obs_ref(tuple(v[chosen] for v in slots),mids[chosen],times[chosen],name)

    _amp_family=HumanoidMAPush._amp_family
    _fetch_conditioned_amp_demo=HumanoidMAPush._fetch_conditioned_amp_demo

    def _reset_boxes(self,ids):
        self._push_hits[ids]=False
        self._push_layout_failed[ids]=False
        super()._reset_boxes(ids)
        primary=self._primary_objects(self._templates[ids])
        retry_scene='source_max_distance' in self.cfg['env']['before']
        try:
            boxes,goals=sample_layout(self._box_states[ids],self._box_size[ids],self._humanoid_root_states[ids,:,:3],
                self._templates[ids]==5,primary,self._active_physical(ids),self._joint['scene'],
                max_attempts=128 if retry_scene else 4096)
        except PushLayoutError as error:
            if not retry_scene:raise
            boxes,goals=error.boxes,error.goals
            self._push_layout_failed[ids[error.pending]]=True
        self._box_states[ids]=boxes;self._push_goals[ids]=goals

    def _reset_task(self,ids):
        super()._reset_task(ids)
        rows,owner=(self._templates[ids]==5).nonzero(as_tuple=True)
        self._tar_pos[ids[rows],owner]=self._push_goals[ids[rows],owner]

    def _ontop_scene_infeasible(self,ids):
        invalid=super()._ontop_scene_infeasible(ids)|self._push_layout_failed[ids]
        primary=self._primary_objects(self._templates[ids]);push=self._templates[ids]==5
        row=torch.arange(len(ids),device=self.device)[:,None]
        boxes=self._box_states[ids];sizes=self._box_size[ids];goals=self._tar_pos[ids]
        invalid |= ~goal_angles_valid(boxes[row,primary],goals,push,self._joint['scene'])
        invalid |= ~paths_valid(boxes,sizes,goals,self._humanoid_root_states[ids,:,:3],push,primary,self._active_physical(ids),self._joint['scene'])
        return invalid

    def _evaluate_ontop_context(self,env_ids=None):
        phi,diag=super()._evaluate_ontop_context(env_ids)
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        bodies=self._rigid_body_pos[ids] if env_ids is None else self._kinematic_humanoid_rigid_body_states[ids,...,:3]
        boxes=self._reward_box_values(self._box_states,ids);sizes=self._reward_box_values(self._box_size,ids)
        hands=bodies[:,:,self._key_body_ids[:2]]
        distance=push_rear_xy(hands,boxes,self._tar_pos[ids],sizes).amax(-1)
        droot=(bodies[:,:,0,:2]-boxes[...,:2]).norm(dim=-1)
        graph=self.relation_runtime.graph;owner=graph.edge_owner[ids];mask=graph.edge_relation[ids]==12
        phi=torch.where(mask,torch.exp(-self._joint['contact']['near_distance_scale']*distance.square()).gather(1,owner),phi)
        progress=1/(1+(droot-self._joint['contact']['progress_delta']).clamp_min(0)/self._joint['contact']['progress_sigma'])
        for key,value in (('progress',progress),('distance',distance),('distance_xy',droot)):
            diag[key]=torch.where(mask,value.gather(1,owner),diag[key])
        diag['z_error']=diag['z_error'].masked_fill(mask,0.)
        return phi,diag

    def pre_physics_step(self,actions):
        super().pre_physics_step(actions);self._push_hits.zero_()

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            _,hit=contact_state(self._rigid_body_pos[:,:,self._key_body_ids[:2]],self._reward_box_values(self._box_states),
                self._tar_pos,self._reward_box_values(self._box_size),self._contact_forces[:,:,self._key_body_ids[:2]],
                self._reward_box_values(self._ontop_box_contact),self._joint['contact'])
            self._push_hits |= hit

    def _compute_reward(self,actions):
        self.relation_runtime.contact.copy_(self._push_hits)
        return super()._compute_reward(actions)

    def _consume_sampling_diagnostics(self):
        push=self._sampling_counts[6].clone();counts=self._solo_counts.clone()
        result=super()._consume_sampling_diagnostics()
        for i,name in enumerate(('holding','sit','climb','at','ontop','push')):
            result['sampling/solo_'+name+'_fraction']=counts[i]/counts.sum().clamp_min(1)
        result['sampling/template_push']=push/(result['sampling/resets']*self.num_agents).clamp_min(1)
        result['sampling/shared_primary_object']=result['sampling/before_pairs']/result['sampling/resets'].clamp_min(1)
        self._solo_counts.zero_()
        return result
