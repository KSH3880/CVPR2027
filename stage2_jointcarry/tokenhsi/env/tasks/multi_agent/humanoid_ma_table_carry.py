"""Training-capable rectangular-table carry, isolated from inference-only payload swaps."""
from pathlib import Path
import json
import numpy as np
import torch
from isaacgym import gymapi
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.joint_carry_reset import JointCarryReset
from utils.joint_carry_rsi import transform_states,rotate_vectors


class TableCarryReset(JointCarryReset):
    def __init__(self,task):
        self.task=task
        spec=task.cfg['env']['tableCarry']
        with np.load(spec['initial_states'],allow_pickle=False) as values:
            self.pool={k:torch.tensor(values[k],device=task.device,dtype=torch.float32)
                       for k in ('root_state','body_state','dof_position','dof_velocity','box_state')}
            self.pool['source_frame']=torch.tensor(values['source_frame'],device=task.device)
        self.source_frame=torch.full((task.num_envs,),-1,device=task.device,dtype=torch.long)
        self.skill_index=torch.full_like(self.source_frame,task._skill.index('carryWith'))
        self.pools={'carryWith':self.pool}

    def reset(self,ids,finalize=True):
        if not len(ids):return
        t=self.task;n=len(ids)
        t._sample_episode_graph(ids);t._sample_box_assignments(ids)
        row=torch.randint(len(self.pool['root_state']),(n,),device=t.device)
        yaw=torch.rand(n,device=t.device)*2*torch.pi;origins=t._env_origins[ids]
        root=self.pool['root_state'][row];body=self.pool['body_state'][row]
        boxes=torch.zeros(n,t.num_objects,13,device=t.device);boxes[...,6]=1
        boxes[...,2]=t._box_size[ids,:,2]/2
        boxes[:,0]=self.pool['box_state'][row]
        boxes[:,1,:2]=boxes.new_tensor([-3.,-2.]);boxes[:,2,:2]=boxes.new_tensor([0.,3.5]);boxes[:,3,:2]=boxes.new_tensor([-3.,2.])
        from utils.joint_carry_goals import sample_goals
        goal=torch.zeros(n,3,device=t.device)
        goal[:,:2]=sample_goals(boxes,t._box_size[ids],body[...,:3],torch.zeros(n,dtype=torch.bool,device=t.device),t.cfg['env']['jointCarryRsi']['goalSampling'])
        goal[:,2]=t._box_size[ids,0,2]/2
        t._humanoid_root_states[ids]=transform_states(root,yaw,origins)
        t._kinematic_humanoid_rigid_body_states[ids]=transform_states(body,yaw,origins)
        t._dof_pos[ids]=self.pool['dof_position'][row];t._dof_vel[ids]=self.pool['dof_velocity'][row]
        t._every_env_init_dof_pos[ids]=self.pool['dof_position'][row]
        t._box_states[ids]=transform_states(boxes,yaw,origins)
        t._tar_pos[ids]=(rotate_vectors(goal,yaw)+origins)[:,None]
        self.source_frame[ids]=self.pool['source_frame'][row]
        self.skill_index[ids]=t._skill.index('carryWith')
        t._stage2_rsi_counts[t._skill.index('carryWith')]+=2*n
        if t._joint_loco_platform is not None:t._joint_loco_platform.reset(ids)
        if not finalize:return
        self.commit(ids);t._refresh_sim_tensors();t._reset_relation_history(ids)
        t._compute_observations(ids);t._compute_amp_observations(ids)
        t._hist_amp_obs_buf[ids]=t._curr_amp_obs_buf[ids,:,None]


class HumanoidMATableCarry(HumanoidMACarry):
    def __init__(self,cfg,*args,**kwargs):
        from utils.table_carry import verify_files
        self.table_spec=cfg['env']['tableCarry'];verify_files(self.table_spec)
        sim_params=kwargs.get('sim_params', args[0] if args else None)
        if sim_params is None or not (sim_params.physx.use_gpu and sim_params.use_gpu_pipeline):
            raise ValueError('Table carry requires GPU PhysX/pipeline')
        from zeroshot.payload import verify_direct_gpu6_context
        verify_direct_gpu6_context(cfg['args'].output_path)
        if cfg['args'].task_graph!='joint_carry_at':
            raise ValueError('Table fine-tune supports two-person joint AT only')
        if cfg['env']['relationReward']['joint_carry'].get('anchor_z')!=self.table_spec['anchor_z']:
            raise ValueError('Table observation and reward anchors differ')
        super().__init__(cfg,*args,**kwargs)
        # This task never samples standalone size-aware box RSI.
        self._size_aware_rsi=False
        self._joint_reset=TableCarryReset(self)
        props=self.gym.get_actor_rigid_body_properties(self.envs[0],self._box_handles[0])
        mass=sum(p.mass for p in props)
        if abs(mass-self.table_spec['mass'])>.001:raise ValueError('Table asset mass mismatch')
        report=dict(size=self._box_size[0,0].tolist(),mass=mass,anchor_z=self.table_spec['anchor_z'],num_envs=self.num_envs,observation_size=self.get_obs_size(),initial_states=len(self._joint_reset.pool['root_state']))
        Path(cfg['args'].output_path,'table_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
        print('TABLE_RUNTIME',json.dumps(report),flush=True)

    def _load_box_asset(self):
        super()._load_box_asset()
        options=gymapi.AssetOptions();options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
        options.angular_damping=.01;options.linear_damping=.01;options.max_angular_velocity=100.
        options.override_com=False;options.override_inertia=False
        path=Path(self.table_spec['asset']).resolve()
        asset=self.gym.load_asset(self.sim,str(path.parent),path.name,options)
        if self.gym.get_asset_rigid_body_count(asset)!=1 or self.gym.get_asset_dof_count(asset)!=0:
            raise ValueError('Table must be one rigid link with zero DOFs')
        for e in range(self.num_envs):self._box_assets[e*self.num_objects]=asset
        self._box_size[:,0]=self._box_size.new_tensor(self.table_spec['size'])
        self._box_scale=self._box_size/self._box_size.new_tensor(self._build_base_size)
