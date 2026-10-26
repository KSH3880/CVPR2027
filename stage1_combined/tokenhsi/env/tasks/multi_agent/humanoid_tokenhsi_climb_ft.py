"""Original TokenHSI CLIMB environment/reset/reward, with our frozen-shape model."""
import json
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_from_angle_axis
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT,NoBoxRsi
from utils.unified_training import append_family
from utils.tokenhsi_climb_reward import compute_climb_reward

class HumanoidTokenHSIClimbFT(HumanoidChairFT):
    def __init__(self,*args,**kwargs):
        self._solo_cache_class=NoBoxRsi
        HumanoidMACarry.__init__(self,*args,**kwargs)
        assert self.num_agents==1 and self.num_objects==1
        self._climb_iet_counts=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
    def _load_box_asset(self):
        from env.tasks.basic_interaction_skills.humanoid_climb import ObjectLib
        root=Path(self.cfg['env']['motion_file']).parent/'objects'
        lib=ObjectLib(self.cfg['env']['chairFT']['asset_split'],str(root),['Box','Cabinet','Table_Square'],self.num_envs,self.device)
        ids=lib._every_env_object_ids
        self._climb_object_lib=lib
        self._box_size=lib._every_env_object_bbox_lengths[:,None]
        self._climb_metadata_heights=lib._every_env_object_tar_climb_pos[:,None,2]
        self._chair_ids=ids[:,None]
        self._chair_kinds=torch.zeros(self.num_envs,1,dtype=torch.long,device=self.device)
        self._chair_features=torch.zeros(self.num_envs,1,6,device=self.device)
        # Route through the same 35D adapter used by the SIT FT; no backrest.
        self._chair_features[...,0]=1
        self._chair_seats=torch.zeros(self.num_envs,1,3,device=self.device)
        self._chair_seat_extents=self._box_size[...,:2].clone()
        self._chair_table_ft=False
        self._box_density=torch.full((self.num_envs,),100.,device=self.device)
        self._box_assets=[];cache={}
        for mid in ids.cpu().tolist():
            if mid not in cache:
                opt=gymapi.AssetOptions();opt.fix_base_link=True;opt.angular_damping=.01;opt.linear_damping=.01;opt.max_angular_velocity=100.;opt.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                opt.use_mesh_materials=True;opt.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX;opt.vhacd_enabled=True;opt.vhacd_params.resolution=100000;opt.vhacd_params.max_convex_hulls=128;opt.vhacd_params.max_num_vertices_per_ch=64;opt.replace_cylinder_with_capsule=False
                cache[mid]=self.gym.load_asset(self.sim,'./',lib._obj_urdfs[mid],opt)
                assert cache[mid] is not None
            self._box_assets.append(cache[mid])
        self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
        self._rsi_asset_sizes=torch.full_like(self._box_size,.45)
    def _init_relation_runtime(self):
        self._size_template_probabilities.zero_();self._size_template_probabilities[...,2]=1
        return HumanoidMACarry._init_relation_runtime(self)
    def _reset_ref_state_init(self,slot_env,slot_agent):
        sk=torch.multinomial(self._skill_init_prob,len(slot_env),replacement=True)
        for uid,name in enumerate(self._skill):
            selected=(sk==uid).nonzero().flatten()
            if not len(selected):continue
            assert name in ('loco','climb')
            rows,owners=slot_env[selected],slot_agent[selected];lib=self._motion_lib[name]
            mid=lib.sample_motions(len(rows));times=lib.sample_time_rsi(mid)
            pos,rot,dof,vel,ang,dvel,_=lib.get_motion_state(mid,times);offset=self._env_origins[rows]+self._agent_spawn_offsets[owners]
            self._set_env_state(rows,owners,pos+offset,rot,dof,vel,ang,dvel)
            bp,bq,bv,ba=lib.get_motion_state_max(mid,times)
            self._kinematic_humanoid_rigid_body_states[rows,owners]=torch.cat([bp+offset[:,None],bq,bv,ba],-1)
            self._every_env_init_dof_pos[rows,owners]=dof
            self._reset_ref_slots[name]=(rows,owners);self._reset_ref_motion_ids[name]=mid;self._reset_ref_motion_times[name]=times
    def _reset_boxes(self,env_ids):
        self._platform_pos[env_ids]=self._platform_default_pos[env_ids]
        for name,slots in self._reset_ref_slots.items():
            rows,owners=slots
            if name=='climb':
                pos,rot=self._motion_lib[name].get_obj_motion_state_single_frame(self._reset_ref_motion_ids[name]);pos=pos.clone();pos[:,:2]+=self._env_origins[rows,:2]+self._agent_spawn_offsets[owners,:2]
            else:
                delta=torch.randn(len(rows),2,device=self.device);delta/=delta.norm(dim=-1,keepdim=True);delta*=1+9*torch.rand(len(rows),1,device=self.device)
                pos=torch.zeros(len(rows),3,device=self.device);pos[:,:2]=self._humanoid_root_states[rows,owners,:2]+delta
                axis=torch.zeros(len(rows),3,device=self.device);axis[:,2]=1
                rot=quat_from_angle_axis(2*torch.pi*torch.rand(len(rows),device=self.device),axis)
            pos[:,2]=self._box_size[rows,0,2]/2
            self._box_states[rows,0,:3]=pos;self._box_states[rows,0,3:7]=rot;self._box_states[rows,0,7:13]=0
    def _ontop_scene_infeasible(self,ids):
        # Original single-agent CLIMB does not add native-cache support/front filters.
        return torch.zeros(len(ids),dtype=torch.bool,device=self.device)
    def _amp_family(self,env_ids=None,slot_agent=None):
        n=self.num_envs if env_ids is None else len(env_ids)
        if slot_agent is None:n*=self.num_agents
        return torch.full((n,),2,dtype=torch.long,device=self.device)
    def _fetch_conditioned_amp_demo(self,num_samples):
        chosen=torch.multinomial(torch.tensor([.3,.4,.3],device=self.device),num_samples,replacement=True)
        result=torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        for i,name in enumerate(['loco','climb','climbNoRSI']):
            rows=(chosen==i).nonzero().flatten()
            if not len(rows):continue
            lib=self._motion_lib[name];mid=lib.sample_motions(len(rows));truncate=self.dt*(self._num_amp_obs_steps-1)
            times=lib.sample_time(mid,truncate_time=truncate)+truncate
            obs=self.build_amp_obs_demo(mid,times,lib).reshape(len(rows),self._num_amp_obs_steps,-1)
            result[rows]=append_family(obs,torch.full((len(rows),self._num_amp_obs_steps),2,dtype=torch.long,device=self.device),3).flatten(1)
        return result
    def _compute_reward(self,actions):
        HumanoidMACarry._compute_reward(self,actions)
        root=self._humanoid_root_states[:,0];obj=self._box_states[:,0]
        target=obj[:,:3].clone();target[:,2]+=self._climb_metadata_heights[:,0]+self._char_h
        radius=self._box_size[:,0,:2].norm(dim=-1)/2+.3
        reward=compute_climb_reward(root[:,:3],self._prev_root_pos[:,0],obj[:,:3],self.dt,target,self._rigid_body_pos[:,0],self._key_body_ids[[2,3]],self._char_h,radius,True,1.,1.3)
        if self._power_reward:reward-=self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum((-1,-2))
        self.rew_buf[:]=reward
        self._climb_iet_counts+=(root[:,:3]-target).norm(dim=-1)<=.2
    def _compute_reset(self):
        from env.tasks.basic_interaction_skills.humanoid_climb import compute_humanoid_reset_with_IET,compute_humanoid_reset
        args=(self.reset_buf,self.progress_buf,self._contact_forces[:,0],self._contact_body_ids,self._rigid_body_pos[:,0],self.max_episode_length,self._enable_early_termination,self._termination_heights)
        if not self._is_eval:
            reset,terminated,_=compute_humanoid_reset_with_IET(*args,60,self._climb_iet_counts)
        else:
            reset,terminated=compute_humanoid_reset(*args)
        self.reset_buf[:]=reset;self._terminate_buf[:]=terminated
    def _reset_envs(self,env_ids):
        self._climb_iet_counts[env_ids]=0
        result=HumanoidMACarry._reset_envs(self,env_ids)
        if len(env_ids)==self.num_envs and not hasattr(self,'_token_climb_validated'):
            assert set(self._motion_lib)==({'loco'} if self._is_eval else {'loco','climb','climbNoRSI'})
            counts={k:len(v[0]) for k,v in self._reset_ref_slots.items()}
            demo=None if self._is_eval else self.fetch_amp_obs_demo(64).reshape(64,self._num_amp_obs_steps,-1)
            if demo is not None:assert torch.isfinite(demo).all() and (demo[...,-3:].argmax(-1)==2).all()
            Path(self.cfg['args'].output_path).mkdir(parents=True,exist_ok=True)
            Path(self.cfg['args'].output_path,'tokenhsi_climb_validation.json').write_text(json.dumps(dict(num_envs=self.num_envs,num_agents=1,num_objects=1,initial_skills=counts,loaded_motion_libraries=list(self._motion_lib),climb_loaded=True,native_cache_used=False,front_filter=False,amp_climb_family_only=True,amp_history_finite=bool(torch.isfinite(self._amp_obs_buf).all()),episode_length=self.max_episode_length),indent=2))
            self._token_climb_validated=True
        return result
