"""Two-person OPEN/PUSH joint learning with per-owner task-conditioned AMP."""
from pathlib import Path
import hashlib
import os
from isaacgym import gymapi, gymtorch
import torch
import yaml
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, compute_agent_collision_penalty
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime, approach_xy_diagnostics
from utils.door_hold import closing_torque, hand_support_contact, support_with_grace, init_posture_bodies, current_posture, collect_foot_support
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.unified_training import append_family
from utils.open_push_lanes import sample_lanes, owner_values


class HumanoidMAOpenPush(HumanoidMACarry):
    REWARD_TERM_NAMES=('state','progress','success','power','collision','total')

    def __init__(self,cfg,*args,**kwargs):
        self._joint_cfg=cfg['env']
        from utils.open_push_spec import scene_layout
        self._scene_layout=scene_layout(cfg['env'])
        self._door_cfg=cfg['env']['door']
        self._hold_spec=cfg['env']['doorHold']
        self._extra_object_dofs=2
        self._amp_family_count=2
        self._joint_rsi_spec=cfg['env'].get('openPushRSI')
        self._joint_rsi_bank=None
        super().__init__(cfg,*args,**kwargs)
        init_posture_bodies(self)
        # Gym simulation tensors are env-local on this runtime, as in Door.
        self._env_origins.zero_()
        self._amp_expert_skills=list(cfg['env']['skill'])
        self._closing_forces=torch.zeros(self.num_envs,self.num_agents*self.num_dof+2,device=self.device)
        self._support_hits=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self._push_hits=torch.zeros_like(self._support_hits)
        self._diag={};self._diag_steps=0
        self._rsi_interaction=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self._rsi_late=torch.zeros_like(self._rsi_interaction)
        self._rsi_phase=torch.zeros(self.num_envs,2,device=self.device)
        self._push_episode_start=self._box_states[:,2:,:3].clone()
        if self._joint_rsi_spec:
            from utils.open_push_rsi import OpenPushRSIBank, bank_contract, geometry_candidates
            proposals=None
            if cfg.get('_rsi_build',False):
                assert self._is_eval and cfg['env']['enableTrackInitState']
                proposals=geometry_candidates(self._motion_lib,self._key_body_ids[:2].cpu().tolist(),
                    self._key_body_ids[2:].cpu().tolist(),self.dt*(self._num_amp_obs_steps-1))
            self._joint_rsi_bank=OpenPushRSIBank(self._joint_rsi_spec,self._motion_lib,
                bank_contract(cfg['env']),proposals,max_open_angle=self._relation_cfg['door']['target_angle']-self._relation_cfg['door']['angle_tolerance']-torch.pi/36)

    def get_open_push_contract(self):
        contract=dict(revision=5,open_reward_contract='rsi_baseline_plus_hand_angle_v1',push_reward_contract='rsi_baseline_plus_hand_distance_midpoint_xy_v2',shuffle_token_order=self._relation_graph_spec.get('shuffle_token_order',False),task_probabilities=self._relation_graph_spec['task_probabilities'],
            manifest_sha256=hashlib.sha256(Path(self.cfg['env']['motion_file']).read_bytes()).hexdigest(),
            skill=self.cfg['env']['skill'],expert_mapping=self.cfg['env']['ampExpertMapping'],
            init=self.cfg['env']['skillInitProb'],door=self._door_cfg,hold=self._hold_spec,
            spawn_distance_range=self.cfg['env']['doorSpawnDistanceRange'],push_box=self.cfg['env']['pushBox'],
            task_labels=['OPEN','PUSH'],object_obs_size=self.get_clean_object_obs_size(),
            share_edge_encoder=yaml.safe_load(Path(self.cfg['args'].cfg_train).read_text())['params']['network']['transformer'].get('share_edge_encoder',False))
        if self._joint_rsi_spec:
            contract['rsi']=self._joint_rsi_spec
            contract['rsi_bank_sha256']=hashlib.sha256(Path(self._joint_rsi_spec['bank_file']).read_bytes()).hexdigest()
        if 'openPushLayout' in self.cfg['env']:
            contract['layout']=dict(self._scene_layout)
        if self.cfg['env'].get('openPushRandomizeLanes',False):
            contract['lane_randomization']={'version':1,'swap_probability':.5}
        if 'pushSceneRandomization' in self.cfg['env']:
            contract['push_scene']=self.cfg['env']['pushSceneRandomization']
        if 'pushGoalRandomization' in self.cfg['env']:
            contract['push_goal']=self.cfg['env']['pushGoalRandomization']
        if 'contact_at' in self._relation_cfg:
            contract['push_reward_contract']='independent_contact_plus_legacy_at_v1'
            contract['contact_at']=dict(self._relation_cfg['contact_at'])
        return contract

    def _build_agent_spawn_offsets(self,device_type,device_id):
        HumanoidMA._build_agent_spawn_offsets(self,device_type,device_id)
        layout=self._scene_layout
        self._agent_spawn_offsets[:]=self._agent_spawn_offsets.new_tensor([
            [layout['door_x']+.78,(a-.5)*layout['lane_spacing']-2.,0.] for a in range(self.num_agents)])

    def _load_box_asset(self):
        size=self.cfg['env']['pushBox']['size']
        self._box_size=torch.tensor([[.9,.06,1.96],[.9,.06,1.96],size,size],device=self.device).unsqueeze(0).repeat(self.num_envs,1,1)
        options=gymapi.AssetOptions();options.fix_base_link=True;options.collapse_fixed_joints=False
        options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
        asset=Path(__file__).resolve().parents[3]/'assets/door'
        self._door_asset=self.gym.load_asset(self.sim,str(asset),'door_hold.urdf',options)
        assert self.gym.get_asset_dof_count(self._door_asset)==1
        self._door_body_count=self.gym.get_asset_rigid_body_count(self._door_asset)
        self._door_leaf_index=self.gym.find_asset_rigid_body_index(self._door_asset,'panel')
        self._door_handle_index=self.gym.find_asset_rigid_body_index(self._door_asset,'handle')
        assert min(self._door_leaf_index,self._door_handle_index)>=0
        options=gymapi.AssetOptions();options.density=self.cfg['env']['pushBox']['mass']/(size[0]*size[1]*size[2])
        options.angular_damping=.01;options.linear_damping=.01
        self._push_asset=self.gym.create_box(self.sim,*size,options)

    def _build_box(self,env_id,object_id,env_ptr):
        a=object_id%2;pose=gymapi.Transform()
        if object_id<2:
            pose.p=gymapi.Vec3(self._scene_layout['door_x'],(a-.5)*self._scene_layout['lane_spacing'],0.)
            handle=self.gym.create_actor(env_ptr,self._door_asset,pose,'door_'+str(a),env_id,0,0)
            props=self.gym.get_actor_dof_properties(env_ptr,handle)
            props['driveMode']=gymapi.DOF_MODE_EFFORT;props['stiffness']=0.
            props['damping']=self._door_cfg['hinge']['damping'];props['friction']=self._door_cfg['hinge']['friction']
            self.gym.set_actor_dof_properties(env_ptr,handle,props)
        else:
            pose.p=gymapi.Vec3(self._scene_layout['push_x'],(a-.5)*self._scene_layout['lane_spacing'],self.cfg['env']['pushBox']['size'][2]/2)
            handle=self.gym.create_actor(env_ptr,self._push_asset,pose,'push_box_'+str(a),env_id,0,0)
            props=self.gym.get_actor_rigid_shape_properties(env_ptr,handle)
            for p in props:p.friction=self.cfg['env']['pushBox']['friction'];p.restitution=0.
            self.gym.set_actor_rigid_shape_properties(env_ptr,handle,props)
        self._box_handles.append(handle)

    def _build_box_bps(self):
        super()._build_box_bps()
        self._box_bps[:,:2]+=self._box_bps.new_tensor([.45,0.,1.02])

    def _build_box_tensors(self):
        super()._build_box_tensors()
        self._door_dof_state=self._dof_state.view(self.num_envs,-1,2)[:,self.num_agents*self.num_dof:]
        start=self.num_agents*self.num_bodies;end=start+2*self._door_body_count
        bodies=self._rigid_body_state.view(self.num_envs,-1,13)
        self._door_bodies=bodies[:,start:end].view(self.num_envs,2,self._door_body_count,13)
        self._door_panel_states=self._door_bodies[:,:,self._door_leaf_index]
        self._door_handle_states=self._door_bodies[:,:,self._door_handle_index]
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs,-1,3)
        self._door_forces=forces[:,start:end].view(self.num_envs,2,self._door_body_count,3)
        self._box_forces=forces[:,end:end+2]

    @property
    def door_hinge_angle(self):return self._door_dof_state[...,0]
    @property
    def door_hinge_velocity(self):return self._door_dof_state[...,1]
    @property
    def door_handle_position(self):return self._door_handle_states[...,:3]

    def _logical_box_values(self,values,env_ids=None):
        if values is self._box_states:
            values=torch.cat((self._door_panel_states,self._box_states[:,2:]),1)
        ids=slice(None) if env_ids is None else env_ids
        return self._gather_entity_slots(values[ids],self._logical_box_order[ids])

    def _owner_values(self,values,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        return owner_values(values[ids],self._agent_lanes[ids])

    def _owner_push_boxes(self,env_ids=None):
        return self._owner_push_values(self._box_states[:,2:],env_ids)

    def _owner_push_values(self,values,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        return owner_values(values[ids],self._push_binding[ids])

    def get_clean_object_obs_size(self):return 40

    def _clean_object_features(self,object_local,env_ids):
        ids=slice(None) if env_ids is None else env_ids
        extra=object_local.new_zeros(*object_local.shape[:-1],10)
        extra[:,:2,:2]=self._owner_values(self._door_dof_state,env_ids)[:,:2]
        panel=self._owner_values(self._door_panel_states,env_ids)[:,:2];inv=panel[...,3:7].clone();inv[...,:3]*=-1
        extra[:,:2,2:5]=rotate(inv,self._owner_values(self.door_handle_position,env_ids)[:,:2]-panel[...,:3])
        runtime=getattr(self,'relation_runtime',None)
        if runtime is not None:
            extra[:,:2,5]=runtime.elapsed[ids,:2]/self._hold_spec['duration']
            extra[:,:2,6]=runtime.contact[ids,:2].float()
            extra[:,:2,7]=runtime.contact_age[ids,:2]/(self._hold_spec['contact_grace']+self.dt)
            extra[:,2:,9]=runtime.push_eligible[ids,:2].float()
        extra[:,:2,8]=1. # distinguishes fixed door articulation from movable boxes
        return torch.cat((object_local,extra),-1)

    def _init_relation_runtime(self):
        self._agent_lanes=(torch.arange(self.num_agents,device=self.device)%2).expand(self.num_envs,self.num_agents).clone()
        self._push_binding=self._agent_lanes.clone()
        self._push_reset_states=self._box_states[:,2:].clone()
        self._push_directions=torch.zeros(self.num_envs,2,3,device=self.device);self._push_directions[...,1]=1
        self._push_goal_offsets=self._push_directions*self._relation_cfg['push']['target_distance']
        self.relation_runtime=OpenPushRuntime(self.num_envs,self._relation_graph_spec,
            self._relation_cfg,self._hold_spec,self.dt,self.device,self.cfg['env']['pushBox']['size'],num_agents=self.num_agents)

    def _sample_tasks(self,env_ids):
        preset=self._task_graph_preset
        if preset=='random_open_push':
            return (torch.rand(len(env_ids),2,device=self.device)>=self._relation_graph_spec['task_probabilities']['OPEN']).long()
        row={'open':[0,0],'push':[1,1],'open_push':[0,1],'push_open':[1,0]}[preset]
        return torch.tensor(row,device=self.device).expand(len(env_ids),2).clone()

    def _sample_box_assignments(self,env_ids):
        self._logical_box_order[env_ids]=torch.cat((self._agent_lanes[env_ids],self._push_binding[env_ids]+2),-1)
        self._agent_box_assignment[env_ids]=torch.where(self.relation_runtime.tasks[env_ids]==1,
            self._push_binding[env_ids]+2,self._agent_lanes[env_ids])

    def _reset_ref_state_init(self,slot_env,slot_agent):
        if self._joint_rsi_bank is None or self._is_eval:
            self._reset_loco_ref_state_init(slot_env,slot_agent)
            return
        selected=torch.rand(len(slot_env),device=self.device)<self._joint_rsi_spec['interaction_probability']
        loco=~selected
        if loco.any():self._reset_loco_ref_state_init(slot_env[loco],slot_agent[loco])
        for family,skill in enumerate(('doorOpen','push')):
            chosen=selected & (self.relation_runtime.tasks[slot_env,slot_agent]==family)
            if chosen.any():
                env,agent=slot_env[chosen],slot_agent[chosen]
                self._apply_joint_rsi(env,agent,skill,self._joint_rsi_bank.sample(skill,len(env)))

    def _apply_joint_rsi(self,env,agent,skill,sample):
        lane=(self._push_binding if skill=='push' else self._agent_lanes)[env,agent]
        from utils.open_push_rsi import align_pose, align_push_pose
        lib=self._motion_lib[skill];mids=sample['motion_id'].long();times=sample['time']
        root,rot,dof,vel,ang,dof_vel,_=lib.get_motion_state(mids,times)
        body,body_rot,body_vel,body_ang=lib.get_motion_state_max(mids,times)
        if skill=='push':
            from utils.open_push_rsi import lower_push_arms
            body,body_rot,body_vel,body_ang,dof,dof_vel=lower_push_arms(
                body,body_rot,body_vel,body_ang,dof,dof_vel,
                lib._motions[0].skeleton_tree.parent_indices.tolist(),
                self._key_body_ids[:2].cpu().tolist(),self._dof_body_ids,self._dof_offsets,
                self._relation_cfg['push']['hand_target_height'])
        angle=sample['angle'] if skill=='doorOpen' else torch.zeros_like(times)
        if skill=='doorOpen':
            self._door_dof_state[env,lane,0]=angle
            self._door_dof_state[env,lane,1]=0.
            self._sync_door_reset_states(env.unique())
            target=self.door_handle_position[env,lane]
        else:
            self._box_states[env,lane+2]=self._push_reset_states[env,lane]
            self._box_states[env,lane+2,:3]+=sample['box_progress'][:,None]*self._push_goal_offsets[env,lane]
        if skill=='doorOpen':
            body,body_rot,body_vel,body_ang,q=align_pose(body,body_rot,body_vel,body_ang,rot,
                target,angle,self._key_body_ids[:2].cpu().tolist(),self._joint_rsi_spec['stand_off'])
        else:
            from env.tasks.multi_agent.open_push_reward import push_hand_targets
            targets=push_hand_targets(self._box_states[env,lane+2][:,None],
                self.cfg['env']['pushBox']['size'],self._relation_cfg['push'])[:,0]
            body,body_rot,body_vel,body_ang,q=align_push_pose(body,body_rot,body_vel,body_ang,
                targets,self._key_body_ids[:2].cpu().tolist(),self._joint_rsi_spec['stand_off'])
        self._set_env_state(env,agent,body[:,0],multiply(q,rot),dof,rotate(q,vel),rotate(q,ang),dof_vel)
        self._kinematic_humanoid_rigid_body_states[env,agent]=torch.cat((body,body_rot,body_vel,body_ang),-1)
        self._every_env_init_dof_pos[env,agent]=dof
        self._reset_ref_slots[skill]=(env,agent)
        self._reset_ref_motion_ids[skill]=mids;self._reset_ref_motion_times[skill]=times
        self._rsi_interaction[env,agent]=True
        self._rsi_late[env,agent]=sample.get('late_requested',torch.zeros_like(times,dtype=torch.bool))
        self._rsi_phase[env,agent]=sample['phase']
        if skill=='push':
            self._push_episode_start[env,agent]=self._box_states[env,lane+2,:3]
            distance=(body[:,self._key_body_ids[:2]]-targets).norm(dim=-1)
            for name,value in (('right_reset_distance',distance[:,0]),('left_reset_distance',distance[:,1])):
                key='push/rsi/'+name+'_sum'
                self._diag[key]=self._diag.get(key,0)+value.sum().detach()
            key='push/rsi/retarget_count'
            self._diag[key]=self._diag.get(key,0)+len(env)

    def _init_amp_obs_ref(self,slots,motion_ids,motion_times,skill_name):
        if skill_name!='push' or not self._joint_rsi_spec:
            return super()._init_amp_obs_ref(slots,motion_ids,motion_times,skill_name)
        # History reflects retargeted reset states; discriminator experts stay raw.
        from utils.open_push_rsi import lower_push_arms
        from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations
        env,agent=slots;steps=self._num_amp_obs_steps-1
        mids=motion_ids[:,None].expand(-1,steps).reshape(-1)
        times=(motion_times[:,None]-self.dt*torch.arange(1,steps+1,device=self.device)).reshape(-1)
        lib=self._motion_lib['push']
        root,rot,dof,vel,ang,dof_vel,_=lib.get_motion_state(mids,times)
        body,body_rot,body_vel,body_ang=lib.get_motion_state_max(mids,times)
        body,body_rot,body_vel,body_ang,dof,dof_vel=lower_push_arms(
            body,body_rot,body_vel,body_ang,dof,dof_vel,
            lib._motions[0].skeleton_tree.parent_indices.tolist(),
            self._key_body_ids[:2].cpu().tolist(),self._dof_body_ids,self._dof_offsets,
            self._relation_cfg['push']['hand_target_height'])
        obs=build_amp_observations(root,rot,vel,ang,dof,dof_vel,body[:,self._key_body_ids],
            self._local_root_obs,self._root_height_obs,self._dof_obs_size,self._dof_offsets)
        labels=torch.ones(len(mids),dtype=torch.long,device=self.device)
        self._hist_amp_obs_buf[env,agent]=append_family(obs,labels,2).reshape(len(env),steps,-1)

    def _reset_loco_ref_state_init(self,slot_env,slot_agent):
        super()._reset_ref_state_init(slot_env,slot_agent)
        root=self._humanoid_root_states[slot_env,slot_agent].clone()
        body=self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent].clone()
        yaw=torch.pi/2-yaw_from_quat(root[:,3:7]);q=torch.zeros_like(root[:,3:7]);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
        lo,hi=self.cfg['env']['doorSpawnDistanceRange'];distance=lo+(hi-lo)*torch.rand(len(slot_env),device=self.device)
        push=self.relation_runtime.tasks[slot_env,slot_agent]==1
        target=root[:,:3].clone()
        target[:,0]=torch.where(push,self._scene_layout['push_x'],self._scene_layout['door_x']+.78)
        lane=self._agent_lanes[slot_env,slot_agent]
        target[:,1]=(lane-.5)*self._scene_layout['lane_spacing']-distance-torch.where(push,self.cfg['env']['pushBox']['size'][1]/2,.1)
        if 'pushSceneRandomization' in self.cfg['env'] and push.any():
            from utils.open_push_scene import sample_loco_positions
            desired=torch.rand(int(push.sum()),device=self.device)*2*torch.pi
            turn=desired-yaw_from_quat(root[push,3:7])
            q[push,2]=torch.sin(turn/2);q[push,3]=torch.cos(turn/2)
            relative=rotate(q[push,None].expand_as(body[push,...,3:7]),body[push,...,:3]-root[push,None,:3])
            boxes=self._box_states[slot_env[push],self._push_binding[slot_env[push],slot_agent[push]]+2]
            target[push]=sample_loco_positions(boxes,relative,root[push,2],self.cfg['env']['pushBox']['size'],
                self.cfg['env']['pushSceneRandomization']['owner_loco_distance_range'],
                goals=(self._push_reset_states[slot_env[push],self._push_binding[slot_env[push],slot_agent[push]],:3]+self._push_goal_offsets[slot_env[push],self._push_binding[slot_env[push],slot_agent[push]]]) if 'pushGoalRandomization' in self.cfg['env'] else None,
                min_goal_distance=self.cfg['env'].get('pushGoalRandomization',{}).get('min_distance',1.))
        expanded=q[:,None].expand_as(body[...,3:7])
        body[...,:3]=rotate(expanded,body[...,:3]-root[:,None,:3])+target[:,None]
        body[...,3:7]=multiply(expanded,body[...,3:7])
        body[...,7:10]=rotate(expanded,body[...,7:10]);body[...,10:13]=rotate(expanded,body[...,10:13])
        self._set_env_state(slot_env,slot_agent,target,multiply(q,root[:,3:7]),self._dof_pos[slot_env,slot_agent].clone(),
            rotate(q,root[:,7:10]),rotate(q,root[:,10:13]),self._dof_vel[slot_env,slot_agent].clone())
        self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent]=body

    def _sync_door_reset_states(self,ids):
        angles=self.door_hinge_angle[ids];panel=self._box_states[ids,:2].clone()
        q=torch.zeros_like(panel[...,3:7]);q[...,2]=torch.sin(angles/2);q[...,3]=torch.cos(angles/2)
        panel[...,3:7]=q;panel[...,7:]=0.;self._door_panel_states[ids]=panel
        handle=panel.clone();offset=panel.new_tensor([.78,-.1,1.]).expand(len(ids),2,3)
        handle[...,:3]+=rotate(q,offset);self._door_handle_states[ids]=handle

    def _reset_boxes(self,env_ids):
        self._box_states[env_ids]=self._initial_box_states[env_ids]
        self._box_states[env_ids,:,7:]=0.
        if 'pushSceneRandomization' in self.cfg['env']:
            from utils.open_push_scene import sample_boxes
            states,direction=sample_boxes(len(env_ids),self.device,self.cfg['env']['pushSceneRandomization'],self.cfg['env']['pushBox']['size'])
            self._box_states[env_ids,2:]=states;self._push_directions[env_ids]=direction
        self._push_reset_states[env_ids]=self._box_states[env_ids,2:]
        self._push_goal_offsets[env_ids]=self._push_directions[env_ids]*self._relation_cfg['push']['target_distance']
        if 'pushGoalRandomization' in self.cfg['env']:
            from utils.open_push_scene import sample_goal_offsets
            self._push_goal_offsets[env_ids]=sample_goal_offsets(self._push_reset_states[env_ids],self.cfg['env']['pushGoalRandomization'])
        lo,hi=self._door_cfg['resetAngleRange'];self._door_dof_state[env_ids,:,0]=lo+(hi-lo)*torch.rand(len(env_ids),2,device=self.device)
        self._door_dof_state[env_ids,:,1]=0.
        self._sync_door_reset_states(env_ids)

    def _reset_task(self,env_ids):
        target=self.relation_runtime.start[env_ids].clone()
        target+=self._owner_push_values(self._push_goal_offsets,env_ids)
        self._tar_pos[env_ids]=torch.where((self.relation_runtime.tasks[env_ids]==1)[...,None],target,self._owner_values(self.door_handle_position,env_ids))

    def _reset_env_tensors(self,env_ids):
        roots=[self._humanoid_actor_ids[env_ids].flatten(),self._box_actor_ids[env_ids].flatten()]
        if self._enable_markers:
            self._marker_pos[env_ids]=self._tar_pos[env_ids];roots.append(self._marker_actor_ids[env_ids].flatten())
        self._joint_reset_root_ids=torch.cat(roots).contiguous()
        self._joint_reset_dof_ids=torch.cat((self._humanoid_actor_ids[env_ids].flatten(),self._box_actor_ids[env_ids,:2].flatten())).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self._root_states),gymtorch.unwrap_tensor(self._joint_reset_root_ids),len(self._joint_reset_root_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self._dof_state),gymtorch.unwrap_tensor(self._joint_reset_dof_ids),len(self._joint_reset_dof_ids))
        self.progress_buf[env_ids]=0;self.reset_buf[env_ids]=0
        flat=self._flat_slot_ids(env_ids);self._terminate_buf[flat]=0
        if self._is_eval:self._success_buf[flat]=0;self._precision_buf[flat]=float('inf')

    def _reset_envs(self,env_ids):
        if not len(env_ids):return
        self._reset_default_slots=None;self._reset_ref_slots={};self._reset_ref_motion_ids={};self._reset_ref_motion_times={}
        if hasattr(self,'_rsi_interaction'):
            self._rsi_interaction[env_ids]=False;self._rsi_late[env_ids]=False;self._rsi_phase[env_ids]=0.
        self._reset_boxes(env_ids)
        mode=os.environ.get('OPEN_PUSH_LANES','random') if self._is_eval else 'random'
        if not self.cfg['env'].get('openPushRandomizeLanes',False):
            mode='fixed'
        self._agent_lanes[env_ids]=sample_lanes(len(env_ids),self.device,
            self.cfg['env'].get('openPushRandomizeLanes',False),mode)
        self._push_binding[env_ids]=(sample_lanes(len(env_ids),self.device,True) if 'pushSceneRandomization' in self.cfg['env'] else self._agent_lanes[env_ids])
        self.relation_runtime.reset(env_ids,self._sample_tasks(env_ids),self._owner_push_boxes(env_ids))
        self._sample_box_assignments(env_ids);self._reset_actors(env_ids);self._reset_task(env_ids)
        self.relation_runtime.reset_open_credit(env_ids,self._owner_values(self.door_hinge_angle,env_ids))
        self.relation_runtime.reset_push_credit(env_ids,self._owner_push_boxes(env_ids),self._tar_pos[env_ids])
        if hasattr(self,'_push_episode_start'):
            self._push_episode_start[env_ids]=self._owner_push_boxes(env_ids)[...,:3]
        self._reset_env_tensors(env_ids);self._sync_door_reset_states(env_ids)
        self._compute_observations(env_ids);self._init_amp_obs(env_ids)
        if self._joint_rsi_spec and hasattr(self,'_rsi_interaction'):
            for family,name in enumerate(('open','push')):
                mask=self.relation_runtime.tasks[env_ids]==family
                for key,value in (('reset_count',mask),('interaction_count',mask & self._rsi_interaction[env_ids]),
                                  ('late_request_count',mask & self._rsi_late[env_ids])):
                    diag_key=name+'/rsi/'+key
                    self._diag[diag_key]=self._diag.get(diag_key,0)+value.sum().detach()
        self._update_joint_colors(env_ids)

    def _update_joint_colors(self,ids):
        if self.viewer is None and not self.cfg['env'].get('video',{}).get('enable',False):return
        for eid in ids.cpu().tolist():
            for o,handle in enumerate(self._box_handles[eid*4:eid*4+4]):
                binding=self._agent_lanes if o<2 else self._push_binding
                owner=int((binding[eid]==o%2).nonzero().item())
                active=self._agent_box_assignment[eid,owner].item()==o
                color=self._agent_color(owner) if active else gymapi.Vec3(.45,.45,.45)
                for b in range(self._door_body_count if o<2 else 1):
                    self.gym.set_rigid_body_color(self.envs[eid],handle,b,gymapi.MESH_VISUAL,color)

    def _amp_family(self,env_ids=None,slot_agent=None):
        if env_ids is None:return self.relation_runtime.tasks.flatten()
        if slot_agent is None:return self.relation_runtime.tasks[env_ids].flatten()
        return self.relation_runtime.tasks[env_ids,slot_agent]

    def _fetch_conditioned_amp_demo(self,num_samples):
        result=torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        probs=self._relation_graph_spec['task_probabilities']
        families=torch.multinomial(torch.tensor([probs['OPEN'],probs['PUSH']],device=self.device),num_samples,replacement=True)
        for family,name in enumerate(('OPEN','PUSH')):
            rows=(families==family).nonzero().flatten()
            if not len(rows):continue
            skills=torch.multinomial(torch.tensor(self.cfg['env']['ampExpertMapping'][name],device=self.device),len(rows),replacement=True)
            for i,skill in enumerate(self.cfg['env']['skill']):
                selected=rows[skills==i]
                if not len(selected):continue
                lib=self._motion_lib[skill];mids=lib.sample_motions(len(selected))
                history=self.dt*(self._num_amp_obs_steps-1);times=lib.sample_time(mids,truncate_time=history)+history
                obs=self.build_amp_obs_demo(mids,times,lib)
                labels=torch.full(obs.shape[:-1],family,device=self.device,dtype=torch.long)
                result[selected]=append_family(obs,labels,2).reshape(len(selected),-1)
        return result

    def pre_physics_step(self,actions):
        super().pre_physics_step(actions)
        self._posture_foot_forces.zero_()
        self._support_hits.zero_();self._push_hits.zero_();self._closing_forces.zero_()
        if not self._pd_control:self._closing_forces[:,:self.num_agents*self.num_dof]=(self.actions*self.motor_efforts.unsqueeze(0)*self.power_scale).reshape(self.num_envs,-1)
        self._closing_forces[:,-2:]=closing_torque(self.door_hinge_angle,self.door_hinge_velocity,self._door_cfg['autoClose'])
        self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self._closing_forces))

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            collect_foot_support(self)
            hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]];forces=self._contact_forces[:,:,self._key_body_ids[:2]]
            leaf=self._door_forces[:,:,[self._door_leaf_index,self._door_handle_index]].sum(-2)
            handles=self._owner_values(self.door_handle_position)
            angles=self._owner_values(self.door_hinge_angle)
            open_hit=hand_support_contact(hands,handles,forces,self._owner_values(leaf),angles,self._hold_spec)
            self._support_hits|=open_hit
            self.relation_runtime.accumulate_open_credit(angles,open_hit,current_posture(self))
            from env.tasks.multi_agent.open_push_reward import push_hand_contact
            boxes=self._owner_push_boxes()
            push_hit=push_hand_contact(hands,forces,boxes,self._owner_push_values(self._box_forces),self.cfg['env']['pushBox']['size'],self._relation_cfg['push'])
            self._push_hits|=push_hit
            self.relation_runtime.accumulate_push_credit(boxes,self._tar_pos,push_hit)

    def _compute_reward(self,actions):
        handles=self._owner_values(self.door_handle_position)
        angles=self._owner_values(self.door_hinge_angle)
        boxes=self._owner_push_boxes()
        runtime=self.relation_runtime;runtime.posture.copy_(current_posture(self));hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]]
        near=(hands-handles[:,:,None]).norm(dim=-1).amin(-1)<=self._hold_spec['hand_distance']
        runtime.contact_age[:],runtime.contact[:]=support_with_grace(runtime.contact_age,self._support_hits & (runtime.tasks==0),near,self.dt,self._hold_spec['contact_grace'])
        runtime.push_contact.copy_(self._push_hits & (runtime.tasks==1))
        # OPEN target follows its handle; PUSH target remains fixed throughout the episode.
        self._tar_pos.copy_(torch.where((runtime.tasks==0)[...,None],handles,self._tar_pos))
        r=runtime.step(self._humanoid_root_states[...,:3],hands,handles,angles,boxes,self._tar_pos)
        power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(r['reward'])
        collision=-self._agent_collision_coeff*compute_agent_collision_penalty(self._humanoid_root_states[...,:3],self._agent_collision_dist) if self._agent_collision_penalty else torch.zeros_like(power)
        total=r['reward']+power+collision;self.rew_buf[:]=total.flatten()
        terms=torch.stack((r['state'],r['progress'],r['bonus'],power,collision,total),-1).reshape(-1,len(self.REWARD_TERM_NAMES))
        self.extras['reward_terms']=terms;self._reward_term_sums+=terms.sum(0);self._reward_term_count+=len(terms)
        self.extras['open_push_tasks']=runtime.tasks.clone()
        self.extras['open_push_agent_lanes']=self._agent_lanes.clone()
        self.extras['open_push_box_binding']=self._push_binding.clone()
        approach = approach_xy_diagnostics(self._humanoid_root_states[..., :3],
            handles, boxes, self._relation_cfg)
        for family,name in enumerate(('open','push')):
            mask=runtime.tasks==family;count=mask.sum().clamp_min(1)
            self._diag[name+'/samples']=self._diag.get(name+'/samples',0)+mask.sum().detach()
            self._diag[name+'/success_sum']=self._diag.get(name+'/success_sum',0)+(runtime.done & mask).sum().detach()
            if family==0:
                for key,value in {'credited_angle_rad':runtime.open_credit,'required_angle_rad':(self._relation_cfg['door']['target_angle']-self._relation_cfg['door']['angle_tolerance']-runtime.open_start_angle).clamp_min(0),'hand_contact':runtime.contact.float(),'eligible':runtime.open_eligible.float(),'hold_seconds':runtime.elapsed}.items():
                    self._diag[name+'/'+key+'_sum']=self._diag.get(name+'/'+key+'_sum',0)+(value*mask).sum().detach()
            metrics = dict(approach[name], manipulation_progress=r['progress'] /
                self._relation_cfg['progress_reward_weight'])
            for key, value in metrics.items():
                total_key = name + '/' + key + '_sum'
                self._diag[total_key] = self._diag.get(total_key, 0) + (value * mask).sum().detach()
        from env.tasks.multi_agent.open_push_reward import push_debug_metrics
        push_metrics=push_debug_metrics(hands,self._contact_forces[:,:,self._key_body_ids[:2]],
            boxes,self._owner_push_values(self._box_forces),runtime.start,self.cfg['env']['pushBox']['size'],
            self._relation_cfg['push'],runtime.push_contact,runtime.push_eligible,self._push_episode_start,
            self._owner_push_values(self._push_directions))
        push_metrics['credited_forward_m']=runtime.push_credit
        push_metrics['required_forward_m']=(runtime.push_reset_distance-self._relation_cfg['push']['goal_tolerance']).clamp_min(self._relation_cfg['push']['min_contact_progress'])
        self.extras['open_push_paired_contact']=runtime.push_contact.clone()
        self.extras['open_push_both_hands_near']=push_metrics['both_hands_near'].bool()
        mask=runtime.tasks==1
        for key,value in push_metrics.items():
            total_key='push/physics/'+key+'_sum'
            self._diag[total_key]=self._diag.get(total_key,0)+(value*mask).sum().detach()
        self._diag_steps+=1

    def _finish_relation_diagnostics(self):pass
    def consume_relation_diagnostics(self):
        result={}
        for name in ('open','push'):
            samples=self._diag.get(name+'/samples',torch.tensor(0.,device=self.device))
            result[name+'/current_success']=self._diag.get(name+'/success_sum',0)/samples.clamp_min(1)
            result[name+'/sample_count']=samples
            for key in (('distance_xy', 'xy_progress', 'manipulation_progress','credited_angle_rad','required_angle_rad','hand_contact','eligible','hold_seconds') if name=='open' else ('distance_xy', 'xy_progress', 'manipulation_progress')):
                result[name+'/'+key] = self._diag.get(name+'/'+key+'_sum', 0) / samples.clamp_min(1)
            if self._joint_rsi_spec:
                for key in ('reset_count','interaction_count','late_request_count'):
                    result[name+'/rsi/'+key]=self._diag.get(name+'/rsi/'+key,torch.tensor(0.,device=self.device))
        samples=self._diag.get('push/samples',torch.tensor(0.,device=self.device))
        for key,value in self._diag.items():
            if key.startswith('push/physics/') and key.endswith('_sum'):
                result[key[:-4]]=value/samples.clamp_min(1)
        if self._joint_rsi_spec:
            count=self._diag.get('push/rsi/retarget_count',0)
            result['push/rsi/retarget_count']=torch.as_tensor(count,device=self.device,dtype=torch.float)
            for name in ('right_reset_distance','left_reset_distance'):
                result['push/rsi/'+name]=torch.as_tensor(self._diag.get('push/rsi/'+name+'_sum',0),device=self.device)/max(1,count)
        self._diag={};self._diag_steps=0
        return result
    def _compute_metrics_evaluation(self):
        self._success_buf.copy_(self.relation_runtime.done.flatten().long())
        self._precision_buf.copy_(self.relation_runtime.error.flatten())
