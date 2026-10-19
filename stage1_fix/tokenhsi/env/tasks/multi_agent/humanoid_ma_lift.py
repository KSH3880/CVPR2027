"""Two independent agents approach, support and lift their own box."""
import copy
import math
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_mul, quat_rotate
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, compute_agent_collision_penalty
from utils.lift_spec import TEMPLATE, validate_config, lift_scores, reward_step, mask_lift_amp


class HumanoidMALift(HumanoidMACarry):
    REWARD_TERM_NAMES=('support_state','support_progress','support_success','lift_state',
                       'lift_progress','lift_success','power','collision','total')

    def __init__(self,cfg,sim_params,physics_engine,device_type,device_id,headless):
        self._lift_cfg=cfg['env']['liftTask']
        self._dual_amp_cfg=cfg['env'].get('liftAmp')
        self._dual_amp=self._dual_amp_cfg is not None
        if self._dual_amp and self._dual_amp_cfg.get('mode')!='teamhoi_dual_v1':
            raise ValueError('Unknown lift AMP mode')
        validate_config(self._lift_cfg)
        if (cfg['env']['numAgents'],cfg['env']['numObjects'])!=(2,2):
            raise ValueError('Lift requires two independent agents and two boxes')
        if cfg['env']['relationGraph'].get('template')!=TEMPLATE or cfg['env'].get('ampTaskConditioning',False):
            raise ValueError('Lift requires support_lift_v1 and unconditioned masked AMP')
        if cfg['env']['stateInit']!='Default':raise ValueError('Lift starts from standing approach, no object RSI')
        super().__init__(cfg,sim_params,physics_engine,device_type,device_id,headless)
        if self._dof_obs_size!=72 or self._num_amp_obs_per_step!=129:
            raise ValueError('Lift mask requires phys_humanoid_v3 native AMP layout')
        expected=[self.gym.find_actor_rigid_body_handle(self.envs[0],self.humanoid_handles_all[0][0],name)
                  for name in ('right_hand','left_hand','right_foot','left_foot')]
        if self._key_body_ids.tolist()!=expected:raise ValueError('Unexpected AMP key-body order')
        self._lift_body_ids=torch.tensor([self.gym.find_actor_rigid_body_handle(
            self.envs[0],self.humanoid_handles_all[0][0],name)
            for name in ('right_hand','left_hand','right_lower_arm','left_lower_arm')],device=self.device)
        self._lift_hold_time=torch.zeros(self.num_envs,2,device=self.device)
        self._lift_achieved=torch.zeros(self.num_envs,2,device=self.device,dtype=torch.bool)
        self._lift_current_success=torch.zeros_like(self._lift_achieved)
        self._lift_initial_rot=torch.zeros(self.num_envs,2,4,device=self.device);self._lift_initial_rot[...,3]=1
        self._lift_scores=None
        if self._dual_amp:
            self._amp_cond_buf=torch.zeros(self.num_envs,2,self._num_amp_obs_steps,device=self.device)
        self._interaction_checkpoint_metadata=dict(task='independent_support_lift_v1',schema=1,
            graph=copy.deepcopy(cfg['env']['relationGraph']),lift=copy.deepcopy(self._lift_cfg),
            amp='native129_mask_shoulders_elbows_hand_positions_v1',
            motions=copy.deepcopy(cfg['env'].get('motion_file')),history_steps=self._num_amp_obs_steps)
        if self._dual_amp:
            self._interaction_checkpoint_metadata.update(schema=2,amp=copy.deepcopy(self._dual_amp_cfg))

    def _build_agent_spawn_offsets(self,device_type,device_id):
        device='cuda:'+str(device_id) if device_type in ('cuda','GPU') else 'cpu'
        lane=self._lift_cfg['lane_spacing']
        self._agent_spawn_offsets=torch.tensor([[-1.8,-lane/2,0.],[-1.8,lane/2,0.]],device=device)

    def _create_envs(self,num_envs,spacing,num_per_row):
        c=self._lift_cfg
        opts=gymapi.AssetOptions();opts.fix_base_link=True
        self._lift_rail_asset=self.gym.create_box(self.sim,c['box_size'][0],c['rail_width'],c['stand_height'],opts)
        super()._create_envs(num_envs,spacing,num_per_row)
        self._env_origins.zero_()  # PhysX state tensors are env-local.

    def _load_box_asset(self):
        c=self._lift_cfg
        self._box_size=torch.tensor(c['box_size'],device=self.device).expand(self.num_envs*2,3).clone()
        self._box_scale=torch.ones_like(self._box_size)
        self._box_density=torch.full((self.num_envs*2,),c['box_mass']/math.prod(c['box_size']),device=self.device)
        opts=gymapi.AssetOptions();opts.density=float(self._box_density[0]);opts.angular_damping=.01;opts.linear_damping=.01
        asset=self.gym.create_box(self.sim,*c['box_size'],opts)
        self._box_assets=[asset]*(self.num_envs*2)
        self._box_size=self._box_size.view(self.num_envs,2,3)
        self._box_scale=self._box_scale.view(self.num_envs,2,3)

    def _build_env(self,env_id,env_ptr,humanoid_asset):
        super()._build_env(env_id,env_ptr,humanoid_asset)
        c=self._lift_cfg
        for a in range(2):
            for sign in (-1,1):
                pose=gymapi.Transform()
                pose.p=gymapi.Vec3(0.,(a-.5)*c['lane_spacing']+sign*(c['box_size'][1]-c['rail_width'])/2,c['stand_height']/2)
                handle=self.gym.create_actor(env_ptr,self._lift_rail_asset,pose,f'lift_rail_{a}_{sign}',env_id,0)
                self.gym.set_rigid_body_color(env_ptr,handle,0,gymapi.MESH_VISUAL,gymapi.Vec3(.35,.35,.35))
        for a in range(2):
            handle=self._box_handles[env_id*2+a]
            shapes=self.gym.get_actor_rigid_shape_properties(env_ptr,handle)
            for prop in shapes:prop.friction=c['box_friction'];prop.restitution=0.
            self.gym.set_actor_rigid_shape_properties(env_ptr,handle,shapes)
            self.gym.set_rigid_body_color(env_ptr,handle,0,gymapi.MESH_VISUAL,self._agent_color(a))

    def _reset_actors(self,env_ids):
        e,a=self._expand_slots(env_ids)
        self._reset_default(e,a)
        old=self._humanoid_root_states[e,a,:3].clone()
        c=self._lift_cfg
        lo,hi=c['start_distance']
        pos=old.clone();pos[:,0]=-(lo+torch.rand(len(e),device=self.device)*(hi-lo))
        pos[:,1]=(a.float()-.5)*c['lane_spacing']+(torch.rand(len(e),device=self.device)-.5)*.3
        yaw=torch.atan2((a.float()-.5)*c['lane_spacing']-pos[:,1],-pos[:,0])
        q=torch.zeros(len(e),4,device=self.device);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
        self._humanoid_root_states[e,a,:3]=pos;self._humanoid_root_states[e,a,3:7]=q
        self._humanoid_root_states[e,a,7:]=0
        kin=self._kinematic_humanoid_rigid_body_states[e,a].clone()
        kin[:,:,:3]=quat_rotate(q[:,None].expand(-1,self.num_bodies,-1).reshape(-1,4),
            (kin[:,:,:3]-old[:,None]).reshape(-1,3)).view(-1,self.num_bodies,3)+pos[:,None]
        kin[:,:,3:7]=quat_mul(q[:,None].expand(-1,self.num_bodies,-1).reshape(-1,4),kin[:,:,3:7].reshape(-1,4)).view(-1,self.num_bodies,4)
        kin[:,:,7:]=0;self._kinematic_humanoid_rigid_body_states[e,a]=kin
        self._reset_default_slots=(e,a)

    def _reset_boxes(self,env_ids):
        c=self._lift_cfg
        states=self._box_states[env_ids].clone();states.zero_()
        states[:,:,1]=torch.tensor([-.5,.5],device=self.device)*c['lane_spacing']
        states[:,:,2]=c['stand_height']+c['box_size'][2]/2+.002
        states[:,:,6]=1
        self._box_states[env_ids]=states
        self._lift_initial_rot[env_ids]=states[:,:,3:7]
        self._lift_hold_time[env_ids]=0;self._lift_achieved[env_ids]=False;self._lift_current_success[env_ids]=False

    def _reset_task(self,env_ids):
        self._tar_pos[env_ids]=self._box_states[env_ids,:,:3]
        self._tar_pos[env_ids,:,2]=self._lift_cfg['target_height']
        if self._enable_markers:self._marker_pos[env_ids]=self._tar_pos[env_ids]

    def _sample_box_assignments(self,env_ids):
        self._agent_box_assignment[env_ids]=torch.arange(2,device=self.device)
        self._logical_box_order[env_ids]=torch.arange(2,device=self.device)

    def _compute_reward(self,actions):
        p=self._rigid_body_pos
        hand=p[:,:,self._lift_body_ids[:2]];elbow=p[:,:,self._lift_body_ids[2:]]
        contact=self._contact_forces[:,:,self._lift_body_ids].norm(dim=-1)
        # Geometry ordering is hand R,L,forearm L,R.
        contact=contact[:,:,[0,1,3,2]]
        scores=lift_scores(self._humanoid_root_states[:,:,:3],hand,elbow,self._box_states,
            self._box_size,contact,self._tar_pos,self._lift_initial_rot,self._lift_cfg)
        terms,elapsed,achieved,success=reward_step(scores,self._lift_hold_time,self._lift_achieved,self.dt,self._lift_cfg)
        self._lift_hold_time.copy_(elapsed);self._lift_achieved.copy_(achieved);self._lift_current_success.copy_(success)
        self._lift_scores=scores
        power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(elapsed)
        collision=-self._agent_collision_coeff*compute_agent_collision_penalty(self._humanoid_root_states[:,:,:3],self._agent_collision_dist)
        total=terms.sum(-1)+power+collision
        self.rew_buf[:]=total.flatten()
        packed=torch.cat((terms,power[...,None],collision[...,None],total[...,None]),-1).reshape(-1,9)
        self.extras['reward_terms']=packed
        self._reward_term_sums+=packed.sum(0);self._reward_term_count+=self.num_envs*2
        self.extras['lift_success']=self._lift_achieved.flatten().clone()
        self.extras['lift_support_state']=scores['support_state_raw'].flatten()
        self.extras['lift_angle_score']=scores['angle_score'].flatten()

    def _compute_metrics_evaluation(self):
        self._success_buf[:]=self._lift_achieved.flatten().long()
        self._precision_buf[:]=(self._box_states[:,:,:3]-self._tar_pos).norm(dim=-1).flatten()

    def _compute_amp_observations(self,env_ids=None):
        super()._compute_amp_observations(env_ids)
        if self._dual_amp:
            from utils.dual_amp import rim_distance
            ids=slice(None) if env_ids is None else env_ids
            roots=(self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[:,:,:,:3])[ids,:,0]
            self._amp_cond_buf[ids,:,0]=rim_distance(roots,self._box_states[ids],self._box_size[ids],self._dual_amp_cfg['rim_samples'])
        elif env_ids is None:self._curr_amp_obs_buf[:]=mask_lift_amp(self._curr_amp_obs_buf)
        else:self._curr_amp_obs_buf[env_ids]=mask_lift_amp(self._curr_amp_obs_buf[env_ids])

    def build_amp_obs_demo(self,motion_ids,motion_times0,motion_lib):
        obs=super().build_amp_obs_demo(motion_ids,motion_times0,motion_lib)
        return obs if self._dual_amp else mask_lift_amp(obs)

    def fetch_amp_obs_demo_full(self,num_samples):
        lib=self._motion_lib['loco']
        ids=lib.sample_motions(num_samples)
        truncate=self.dt*(self._num_amp_obs_steps-1)
        times=lib.sample_time(ids,truncate_time=truncate)+truncate
        return HumanoidMACarry.build_amp_obs_demo(self,ids,times,lib).reshape(num_samples,-1)

    def _init_amp_obs(self,env_ids):
        super()._init_amp_obs(env_ids)
        if self._dual_amp:
            self._amp_cond_buf[env_ids]=self._amp_cond_buf[env_ids,:,0,None].expand(-1,-1,self._num_amp_obs_steps)

    def _update_hist_amp_obs(self,env_ids=None):
        super()._update_hist_amp_obs(env_ids)
        if self._dual_amp:
            ids=slice(None) if env_ids is None else env_ids
            history=self._amp_cond_buf[ids].clone()
            self._amp_cond_buf[ids,:,1:]=history[:,:,:-1]

    def post_physics_step(self):
        super().post_physics_step()
        if self._dual_amp:
            self.extras['amp_cond']=self._amp_cond_buf.reshape(self.num_envs*2,-1).clone()

