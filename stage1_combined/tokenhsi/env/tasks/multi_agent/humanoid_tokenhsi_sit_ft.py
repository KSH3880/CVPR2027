"""Original TokenHSI SIT environment/reset/reward, with our frozen-shape model."""
import json
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate,quat_from_angle_axis
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT,NoBoxRsi
from utils.unified_training import append_family
from utils.tokenhsi_sit_reward import compute_sit_reward

class HumanoidTokenHSISitFT(HumanoidChairFT):
    def __init__(self,*args,**kwargs):
        self._solo_cache_class=NoBoxRsi
        HumanoidMACarry.__init__(self,*args,**kwargs)
        assert self.num_agents==1 and self.num_objects==1
        self._sit_iet_counts=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self._sit_prev_rot=self._humanoid_root_states[...,3:7].clone()
    def _load_box_asset(self):
        root=Path(__file__).resolve().parents[4]/'chair_assets';split=self.cfg['env']['chairFT']['asset_split']
        catalog=json.loads((root/'catalog.json').read_text())[split];n=self.num_envs
        ids=torch.multinomial(torch.full((len(catalog),),1/len(catalog),device=self.device),n,replacement=True)
        self._chair_ids=ids[:,None];self._chair_kinds=torch.full((n,1),2,device=self.device,dtype=torch.long)
        def array(key):return torch.tensor([x[key] for x in catalog],dtype=torch.float32,device=self.device)[ids,None]
        self._box_size=array('bbox');self._chair_features=array('feature');self._chair_seats=array('seat');self._chair_seat_extents=array('seat_extent')
        self._climb_metadata_heights=torch.zeros(n,1,device=self.device);self._chair_table_ft=False
        self._box_density=torch.full((n,),100.,device=self.device);self._box_assets=[];cache={}
        for mid in ids.cpu().tolist():
            if mid not in cache:
                opt=gymapi.AssetOptions();opt.fix_base_link=True;opt.angular_damping=.01;opt.linear_damping=.01;opt.max_angular_velocity=100.;opt.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                opt.use_mesh_materials=True;opt.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX;opt.vhacd_enabled=True;opt.vhacd_params.resolution=100000;opt.vhacd_params.max_convex_hulls=128;opt.vhacd_params.max_num_vertices_per_ch=64;opt.replace_cylinder_with_capsule=False
                cache[mid]=self.gym.load_asset(self.sim,catalog[mid]['path'],'asset.urdf',opt)
                assert cache[mid] is not None
            self._box_assets.append(cache[mid])
        self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
        self._rsi_asset_sizes=torch.full_like(self._box_size,.45)
    def _init_relation_runtime(self):
        self._size_template_probabilities.zero_();self._size_template_probabilities[...,1]=1
        return HumanoidMACarry._init_relation_runtime(self)
    def _reset_ref_state_init(self,slot_env,slot_agent):
        sk=torch.multinomial(self._skill_init_prob,len(slot_env),replacement=True)
        for uid,name in enumerate(self._skill):
            selected=(sk==uid).nonzero().flatten()
            if not len(selected):continue
            assert name in ('loco','sit')
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
            if name=='sit':
                pos,rot=self._motion_lib[name].get_obj_motion_state_single_frame(self._reset_ref_motion_ids[name]);pos=pos.clone();pos[:,:2]+=self._env_origins[rows,:2]+self._agent_spawn_offsets[owners,:2]
            else:
                delta=torch.randn(len(rows),2,device=self.device);delta/=delta.norm(dim=-1,keepdim=True);delta*=1+4*torch.rand(len(rows),1,device=self.device)
                pos=torch.zeros(len(rows),3,device=self.device);pos[:,:2]=self._humanoid_root_states[rows,owners,:2]+delta
                axis=torch.zeros(len(rows),3,device=self.device);axis[:,2]=1
                rot=quat_from_angle_axis(2*torch.pi*torch.rand(len(rows),device=self.device),axis)
            pos[:,2]=self._box_size[rows,0,2]/2
            self._box_states[rows,0,:3]=pos;self._box_states[rows,0,3:7]=rot;self._box_states[rows,0,7:13]=0
    def _ontop_scene_infeasible(self,ids):
        # Original single-agent SIT does not add native-cache support/front filters.
        return torch.zeros(len(ids),dtype=torch.bool,device=self.device)
    def _amp_family(self,env_ids=None,slot_agent=None):
        n=self.num_envs if env_ids is None else len(env_ids)
        if slot_agent is None:n*=self.num_agents
        return torch.ones(n,dtype=torch.long,device=self.device)
    def _fetch_conditioned_amp_demo(self,num_samples):
        chosen=torch.multinomial(torch.tensor([1/3,2/3],device=self.device),num_samples,replacement=True)
        result=torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        for i,name in enumerate(['loco','sit']):
            rows=(chosen==i).nonzero().flatten()
            if not len(rows):continue
            lib=self._motion_lib[name];mid=lib.sample_motions(len(rows));truncate=self.dt*(self._num_amp_obs_steps-1)
            times=lib.sample_time(mid,truncate_time=truncate)+truncate
            obs=self.build_amp_obs_demo(mid,times,lib).reshape(len(rows),self._num_amp_obs_steps,-1)
            result[rows]=append_family(obs,torch.ones(len(rows),self._num_amp_obs_steps,dtype=torch.long,device=self.device),3).flatten(1)
        return result
    def pre_physics_step(self,actions):
        self._sit_prev_rot[:]=self._humanoid_root_states[...,3:7]
        return super().pre_physics_step(actions)
    def _compute_reward(self,actions):
        super()._compute_reward(actions)  # preserve relation diagnostics only
        root=self._humanoid_root_states[:,0];obj=self._box_states[:,0]
        target=obj[:,:3]+quat_rotate(obj[:,3:7],self._chair_seats[:,0]);target[:,2]+=.1
        reward=compute_sit_reward(root[:,:3],self._prev_root_pos[:,0],root[:,3:7],self._sit_prev_rot[:,0],obj[:,:3],target,1.5,self.dt,True,.5,1.3,.5,3.)
        if self._power_reward:reward-=self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum((-1,-2))
        self.rew_buf[:]=reward
        self._sit_iet_counts+=(root[:,:3]-target).norm(dim=-1)<=.2
    def _compute_reset(self):
        super()._compute_reset()
        if not self._is_eval:self.reset_buf|=(self._sit_iet_counts>=59).long()
    def _reset_envs(self,env_ids):
        self._sit_iet_counts[env_ids]=0
        result=HumanoidMACarry._reset_envs(self,env_ids)
        if len(env_ids)==self.num_envs and not hasattr(self,'_token_sit_validated'):
            assert set(self._motion_lib)==({'loco'} if self._is_eval else {'loco','sit'})
            counts={k:len(v[0]) for k,v in self._reset_ref_slots.items()}
            demo=None if self._is_eval else self.fetch_amp_obs_demo(64).reshape(64,self._num_amp_obs_steps,-1)
            if demo is not None:assert torch.isfinite(demo).all() and (demo[...,-3:].argmax(-1)==1).all()
            Path(self.cfg['args'].output_path).mkdir(parents=True,exist_ok=True)
            Path(self.cfg['args'].output_path,'tokenhsi_sit_validation.json').write_text(json.dumps(dict(num_envs=self.num_envs,num_agents=1,num_objects=1,initial_skills=counts,loaded_motion_libraries=list(self._motion_lib),climb_loaded=False,native_cache_used=False,front_filter=False,amp_sit_family_only=True,amp_history_finite=bool(torch.isfinite(self._amp_obs_buf).all()),episode_length=self.max_episode_length),indent=2))
            self._token_sit_validated=True
        return result
