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


class HumanoidMAOpenPush(HumanoidMACarry):
    REWARD_TERM_NAMES=('state','progress','success','power','collision','total')

    def __init__(self,cfg,*args,**kwargs):
        from utils.push_view_geometry import viewer_geometry
        # Keep the source checkpoint/RSI contract strict. Only the explicitly
        # requested viewer assets and scene clearance use the new geometry.
        self._viewer_geometry_env=viewer_geometry(cfg['env'],cfg['args'],os.environ)
        if self._viewer_geometry_env is not cfg['env']:
            print('[viewer geometry] source checkpoint contract retained; actual PUSH box:',
                  self._viewer_geometry_env['pushBox'], 'path separation:',
                  self._viewer_geometry_env['pushScene']['path_separation'])
        self._joint_cfg=cfg['env']
        self._push_only=cfg['env']['relationGraph'].get('push_only',False)
        self._push_object_obs_size=cfg['env'].get('pushObjectObsSize',30)
        if type(self._push_object_obs_size) is not int or self._push_object_obs_size not in (30,40):
            raise ValueError('pushObjectObsSize must be 30 or 40')
        self._push_slot_start=0 if self._push_only else 2
        from utils.push_scene import validate_scene
        validate_scene(cfg['env'])
        self._push_scene_spec=cfg['env'].get('pushScene')
        self._random_at_goal=bool(self._push_scene_spec and self._push_scene_spec['mode']=='owner_loco_random_at')
        zone_spec=cfg['env'].get('zoneAssignment', {})
        assert set(zone_spec) <= {'swap_probability'}, 'Unknown zoneAssignment option'
        self._zone_swap_probability=float(zone_spec.get('swap_probability', 0.))
        assert 0. <= self._zone_swap_probability <= 1.
        self._door_cfg=cfg['env']['door']
        self._hold_spec=cfg['env']['doorHold']
        self._extra_object_dofs=0 if self._push_only else 2
        self._amp_family_count=2
        self._joint_rsi_spec=cfg['env'].get('openPushRSI')
        self._joint_rsi_bank=None
        self._manipulation_spec=cfg['env'].get('manipulationContact')
        super().__init__(cfg,*args,**kwargs)
        init_posture_bodies(self)
        # Gym simulation tensors are env-local on this runtime, as in Door.
        self._env_origins.zero_()
        self._amp_expert_skills=list(cfg['env']['skill'])
        self._closing_forces=torch.zeros(self.num_envs,2*self.num_dof+self._extra_object_dofs,device=self.device)
        self._support_hits=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self._push_hits=torch.zeros_like(self._support_hits)
        self._diag={};self._diag_steps=0
        self._rsi_interaction=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self._rsi_late=torch.zeros_like(self._rsi_interaction)
        self._rsi_phase=torch.zeros(self.num_envs,2,device=self.device)
        self._push_episode_start=self._box_states[:,self._push_slot_start:,:3].clone()
        if self._manipulation_spec is not None:
            from utils.manipulation_contact import collision_samples
            asset=Path(cfg['env']['asset']['assetRoot'])/cfg['env']['asset']['assetFileName']
            samples=collision_samples(asset)
            names=self.gym.get_actor_rigid_body_names(self.envs[0],self.humanoid_handles_all[0][0])
            self._nonhand_ids=torch.tensor([names.index(name) for name,_,_ in samples],device=self.device)
            self._nonhand_offsets=torch.tensor([point for _,point,_ in samples],device=self.device)
            self._nonhand_radii=torch.tensor([radius for _,_,radius in samples],device=self.device)
            self._body_hits=torch.zeros_like(self._support_hits)
            self._body_force_peak=torch.zeros_like(self.relation_runtime.elapsed)
            self._clean_open_hits=torch.zeros_like(self._support_hits)
            self._clean_push_hits=torch.zeros_like(self._support_hits)
        if self._joint_rsi_spec:
            from utils.open_push_rsi import OpenPushRSIBank, bank_contract, geometry_candidates
            proposals=None
            if cfg.get('_rsi_build',False):
                assert self._is_eval and cfg['env']['enableTrackInitState']
                proposals=geometry_candidates(self._motion_lib,self._key_body_ids[:2].cpu().tolist(),
                    self._key_body_ids[2:].cpu().tolist(),self.dt*(self._num_amp_obs_steps-1))
            if self._joint_rsi_spec.get('pose_mode')=='native_phase':
                from utils.native_push_rsi import NativePushBank
                self._joint_rsi_bank=NativePushBank(self._joint_rsi_spec,self._motion_lib,cfg['env'])
            else:
                self._joint_rsi_bank=OpenPushRSIBank(self._joint_rsi_spec,self._motion_lib,
                    bank_contract(cfg['env']),proposals)

    def get_open_push_contract(self):
        contract=dict(revision=1,shuffle_token_order=self._relation_graph_spec.get('shuffle_token_order',False),task_probabilities=self._relation_graph_spec['task_probabilities'],
            manifest_sha256=hashlib.sha256(Path(self.cfg['env']['motion_file']).read_bytes()).hexdigest(),
            skill=self.cfg['env']['skill'],expert_mapping=self.cfg['env']['ampExpertMapping'],
            init=self.cfg['env']['skillInitProb'],door=self._door_cfg,hold=self._hold_spec,
            spawn_distance_range=self.cfg['env']['doorSpawnDistanceRange'],push_box=self.cfg['env']['pushBox'],
            task_labels=['OPEN','PUSH'],object_obs_size=self.get_clean_object_obs_size(),
            share_edge_encoder=yaml.safe_load(Path(self.cfg['args'].cfg_train).read_text())['params']['network']['transformer'].get('share_edge_encoder',False))
        if self._joint_rsi_spec:
            contract['rsi']=self._joint_rsi_spec
            contract['rsi_bank_sha256']=hashlib.sha256(Path(self._joint_rsi_spec['bank_file']).read_bytes()).hexdigest()
        if self._manipulation_spec is not None:
            contract['manipulation_contact']=dict(revision=1,parameters=self._manipulation_spec,
                asset_sha256=hashlib.sha256((Path(self.cfg['env']['asset']['assetRoot'])/
                    self.cfg['env']['asset']['assetFileName']).read_bytes()).hexdigest())
        if self._relation_cfg['push'].get('control_mode'):
            contract['push_control']=dict(revision=1,parameters=self._relation_cfg['push'])
        if self._push_scene_spec:
            contract['push_scene']=dict(revision=1,parameters=self._push_scene_spec,
                box_reset=self.cfg['env']['box']['reset'])
        if self._zone_swap_probability:
            contract['zone_assignment']=dict(revision=1,swap_probability=self._zone_swap_probability)
        if self._push_only:contract['push_only']=dict(revision=1,num_objects=2,entity_tokens=6)
        if self._relation_cfg.get('edge_aggregation'):
            contract['push_at_reward']=dict(revision=1,parameters=self._relation_cfg)
        if self._relation_cfg.get('push_at',{}).get('hand_reference')=='goal_opposite_surface_xy':
            contract['push_rsi_alignment']='goal_opposite_surface_xy_v1'
        if self._relation_cfg.get('contact_at'):
            contract['push_rsi_alignment']='goal_contact_face_xy_v1'
        return contract

    def _build_agent_spawn_offsets(self,device_type,device_id):
        HumanoidMA._build_agent_spawn_offsets(self,device_type,device_id)
        self._agent_spawn_offsets[:]=self._agent_spawn_offsets.new_tensor([[1.78,-4.5,0.],[1.78,.5,0.]])
        if self._push_scene_spec:
            self._agent_spawn_offsets[:,:2]=self._agent_spawn_offsets.new_tensor(self._push_scene_spec['owner_anchor_xy'])

    def _load_box_asset(self):
        size=self.cfg['env']['pushBox']['size']
        if self._push_only and self.cfg['env']['pushBox'].get('size_profiles'):
            profiles=self._viewer_geometry_env['pushBox']['size_profiles']
            choices=torch.randint(len(profiles),(self.num_envs,2),device=self.device)
            if '_native_profile' in self.cfg:
                choices[:]=self.cfg['_native_profile']
            elif self.cfg.get('_native_all_profiles'):
                choices[:]=torch.arange(self.num_envs,device=self.device)[:,None]%len(profiles)
            self._box_size=torch.tensor(profiles,device=self.device)[choices]
            self._push_assets=[]
            for dims in profiles:
                options=gymapi.AssetOptions();options.density=self.cfg['env']['pushBox']['mass']/(dims[0]*dims[1]*dims[2])
                options.angular_damping=.01;options.linear_damping=.01
                self._push_assets.append(self.gym.create_box(self.sim,*dims,options))
            self._push_profile_ids=choices
            self._door_body_count=0
            return
        if self._push_only:
            self._box_size=torch.tensor([size,size],device=self.device).unsqueeze(0).repeat(self.num_envs,1,1)
            options=gymapi.AssetOptions();options.density=self.cfg['env']['pushBox']['mass']/(size[0]*size[1]*size[2])
            options.angular_damping=.01;options.linear_damping=.01
            self._push_asset=self.gym.create_box(self.sim,*size,options)
            self._door_body_count=0
            return
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
        if object_id<2 and not self._push_only:
            pose.p=gymapi.Vec3(1.,(a-.5)*5.,0.)
            handle=self.gym.create_actor(env_ptr,self._door_asset,pose,'door_'+str(a),env_id,0,0)
            props=self.gym.get_actor_dof_properties(env_ptr,handle)
            props['driveMode']=gymapi.DOF_MODE_EFFORT;props['stiffness']=0.
            props['damping']=self._door_cfg['hinge']['damping'];props['friction']=self._door_cfg['hinge']['friction']
            self.gym.set_actor_dof_properties(env_ptr,handle,props)
        else:
            pose.p=gymapi.Vec3(-1.5,(a-.5)*5.,self.cfg['env']['pushBox']['size'][2]/2)
            asset=self._push_assets[int(self._push_profile_ids[env_id,a])] if hasattr(self,'_push_assets') else self._push_asset
            handle=self.gym.create_actor(env_ptr,asset,pose,'push_box_'+str(a),env_id,0,0)
            props=self.gym.get_actor_rigid_shape_properties(env_ptr,handle)
            for p in props:p.friction=self.cfg['env']['pushBox']['friction'];p.restitution=0.
            self.gym.set_actor_rigid_shape_properties(env_ptr,handle,props)
        self._box_handles.append(handle)

    def _build_box_bps(self):
        super()._build_box_bps()
        if not self._push_only:self._box_bps[:,:2]+=self._box_bps.new_tensor([.45,0.,1.02])

    def _build_box_tensors(self):
        super()._build_box_tensors()
        if self._push_only:
            forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs,-1,3)
            self._box_forces=forces[:,2*self.num_bodies:2*self.num_bodies+2]
            self._no_door_angles=torch.zeros(self.num_envs,2,device=self.device)
            self._no_door_handles=torch.zeros(self.num_envs,2,3,device=self.device)
            return
        self._door_dof_state=self._dof_state.view(self.num_envs,-1,2)[:,2*self.num_dof:]
        start=2*self.num_bodies;end=start+2*self._door_body_count
        bodies=self._rigid_body_state.view(self.num_envs,-1,13)
        self._door_bodies=bodies[:,start:end].view(self.num_envs,2,self._door_body_count,13)
        self._door_panel_states=self._door_bodies[:,:,self._door_leaf_index]
        self._door_handle_states=self._door_bodies[:,:,self._door_handle_index]
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs,-1,3)
        self._door_forces=forces[:,start:end].view(self.num_envs,2,self._door_body_count,3)
        self._box_forces=forces[:,end:end+2]

    def _build_assignment_tensors(self):
        super()._build_assignment_tensors()
        self._agent_zone=torch.arange(2,device=self.device).expand(self.num_envs,2).clone()
        if self._push_scene_spec:
            self._scene_generator=torch.Generator(device=self.device)
            self._scene_generator.manual_seed(torch.initial_seed()+209759)
            self._push_reset_states=torch.zeros(self.num_envs,2,13,device=self.device)
            self._push_reset_states[...,6]=1.
            self._push_goal_direction=torch.zeros(self.num_envs,2,3,device=self.device)
            self._push_goal_direction[...,1]=1.
            if self._random_at_goal:
                self._push_random_goals=torch.zeros(self.num_envs,2,3,device=self.device)
                self._push_goal_distance=torch.zeros(self.num_envs,2,device=self.device)
        if self._zone_swap_probability:
            # Zone sampling has its own stream; AMP/RSI probabilities stay unchanged.
            self._zone_generator=torch.Generator(device=self.device)
            self._zone_generator.manual_seed(torch.initial_seed()+104729)

    def _zone_values(self,values,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        if not self._zone_swap_probability:return values[ids]
        return self._gather_entity_slots(values[ids],self._agent_zone[ids])

    def _push_sizes(self,env_ids=None):
        if not self.cfg['env']['pushBox'].get('size_profiles'):
            return self.cfg['env']['pushBox']['size']
        return self._zone_values(self._box_size[:,self._push_slot_start:],env_ids)

    def _owned_push_states(self,env_ids=None):
        return self._zone_values(self._box_states[:,self._push_slot_start:],env_ids)

    def _sample_zone_assignments(self,env_ids):
        identity=torch.arange(2,device=self.device).expand(len(env_ids),2)
        if not self._zone_swap_probability:return identity
        swap=torch.rand(len(env_ids),device=self.device,generator=self._zone_generator)<self._zone_swap_probability
        return torch.where(swap[:,None],1-identity,identity)

    @property
    def door_hinge_angle(self):return self._no_door_angles if self._push_only else self._zone_values(self._door_dof_state)[...,0]
    @property
    def door_hinge_velocity(self):return self._no_door_angles if self._push_only else self._zone_values(self._door_dof_state)[...,1]
    @property
    def door_handle_position(self):return self._no_door_handles if self._push_only else self._zone_values(self._door_handle_states)[...,:3]

    def _logical_box_values(self,values,env_ids=None):
        if values is self._box_states and not self._push_only:
            values=torch.cat((self._door_panel_states,self._box_states[:,self._push_slot_start:]),1)
        ids=slice(None) if env_ids is None else env_ids
        if not self._zone_swap_probability:return values[ids]
        return self._gather_entity_slots(values[ids],self._logical_box_order[ids])

    def get_clean_object_obs_size(self):
        return getattr(self,'_push_object_obs_size',30) if self._push_only else 40

    def _clean_object_features(self,object_local,env_ids):
        ids=slice(None) if env_ids is None else env_ids
        if self._push_only and self.get_clean_object_obs_size()==30:
            # Door-free policies observe box physics only; manipulation credit
            # and contact history remain internal to rewards and diagnostics.
            return object_local
        extra=object_local.new_zeros(*object_local.shape[:-1],10)
        if self._push_only:
            runtime=getattr(self,'relation_runtime',None)
            if runtime is not None:
                extra[:,:,9]=runtime.push_eligible[ids].float()
                extra[:,:,0]=runtime.push_credited[ids]/self._relation_cfg['push']['target_distance']
                extra[:,:,1]=(runtime.push_hand_distance[ids]/self._relation_cfg['push']['min_contact_progress']).clamp(max=1)
                extra[:,:,2]=runtime.push_completed[ids].float()
                extra[:,:,3]=runtime.body_contact[ids].float()
            return torch.cat((object_local,extra),-1)
        extra[:,:2,:2]=self._zone_values(self._door_dof_state,env_ids)
        panel=self._zone_values(self._door_panel_states,env_ids);inv=panel[...,3:7].clone();inv[...,:3]*=-1
        extra[:,:2,2:5]=rotate(inv,self.door_handle_position[ids]-panel[...,:3])
        runtime=getattr(self,'relation_runtime',None)
        if runtime is not None:
            extra[:,:2,5]=runtime.elapsed[ids]/self._hold_spec['duration']
            extra[:,:2,6]=runtime.contact[ids].float()
            extra[:,:2,7]=runtime.contact_age[ids]/(self._hold_spec['contact_grace']+self.dt)
            extra[:,2:,9]=runtime.push_eligible[ids].float()
            if self._manipulation_spec is not None:
                extra[:,:2,9]=runtime.open_credited[ids]/self._relation_cfg['door']['target_angle']
                extra[:,2:,0]=runtime.push_credited[ids]/self._relation_cfg['push']['target_distance']
                extra[:,2:,1]=(runtime.push_hand_distance[ids]/self._relation_cfg['push']['min_contact_progress']).clamp(max=1)
                extra[:,2:,2]=runtime.push_completed[ids].float()
                extra[:,2:,3]=runtime.body_contact[ids].float()
        extra[:,:2,8]=1. # distinguishes fixed door articulation from movable boxes
        return torch.cat((object_local,extra),-1)

    def _init_relation_runtime(self):
        self.relation_runtime=OpenPushRuntime(self.num_envs,self._relation_graph_spec,
            self._relation_cfg,self._hold_spec,self.dt,self.device,self._push_sizes(),self._manipulation_spec)

    def _sample_tasks(self,env_ids):
        preset=self._task_graph_preset
        if preset=='random_open_push':
            return (torch.rand(len(env_ids),2,device=self.device)>=self._relation_graph_spec['task_probabilities']['OPEN']).long()
        row={'open':[0,0],'push':[1,1],'open_push':[0,1],'push_open':[1,0]}[preset]
        return torch.tensor(row,device=self.device).expand(len(env_ids),2).clone()

    def _sample_box_assignments(self,env_ids):
        self._agent_zone[env_ids]=self._sample_zone_assignments(env_ids)
        zones=self._agent_zone[env_ids]
        self._logical_box_order[env_ids]=zones if self._push_only else torch.cat((zones,zones+2),-1)
        self._agent_box_assignment[env_ids]=zones if self._push_only else zones+2*self.relation_runtime.tasks[env_ids]

    def _reset_ref_state_init(self,slot_env,slot_agent):
        if self._joint_rsi_bank is None or (self._is_eval and os.environ.get('TOKENHSI_EVAL_TRAIN_RSI') != '1'):
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
        from utils.open_push_rsi import align_pose, align_push_pose
        if self._push_only and skill!='push':raise ValueError('Door-free RSI accepts PUSH only')
        zone=self._agent_zone[env,agent]
        lib=self._motion_lib[skill];mids=sample['motion_id'].long();times=sample['time']
        root,rot,dof,vel,ang,dof_vel,_=lib.get_motion_state(mids,times)
        body,body_rot,body_vel,body_ang=lib.get_motion_state_max(mids,times)
        if skill=='push' and self._joint_rsi_spec.get('pose_mode')!='native_phase':
            from utils.open_push_rsi import lower_push_arms
            body,body_rot,body_vel,body_ang,dof,dof_vel=lower_push_arms(
                body,body_rot,body_vel,body_ang,dof,dof_vel,
                lib._motions[0].skeleton_tree.parent_indices.tolist(),
                self._key_body_ids[:2].cpu().tolist(),self._dof_body_ids,self._dof_offsets,
                self._relation_cfg['push']['hand_target_height'])
        angle=sample['angle'] if skill=='doorOpen' else torch.zeros_like(times)
        if skill=='doorOpen':
            self._door_dof_state[env,zone,0]=angle
            self._door_dof_state[env,zone,1]=0.
            self._sync_door_reset_states(env.unique())
            target=self.door_handle_position[env,agent]
        else:
            if self._push_scene_spec:
                self._box_states[env,zone+self._push_slot_start]=self._push_reset_states[env,zone]
                self._box_states[env,zone+self._push_slot_start,:3]+=self._push_goal_direction[env,agent]*(
                    sample['box_progress']*(self._push_goal_distance[env,agent] if self._random_at_goal else self._relation_cfg['push']['target_distance']))[:,None]
            else:
                self._box_states[env,zone+self._push_slot_start]=self._initial_box_states[env,zone+self._push_slot_start]
                self._box_states[env,zone+self._push_slot_start,1]+=sample['box_progress']*self._relation_cfg['push']['target_distance']
        if skill=='doorOpen':
            body,body_rot,body_vel,body_ang,q=align_pose(body,body_rot,body_vel,body_ang,rot,
                target,angle,self._key_body_ids[:2].cpu().tolist(),self._joint_rsi_spec['stand_off'])
        else:
            from env.tasks.multi_agent.open_push_reward import push_rsi_hand_targets
            # _reset_task runs after RSI; use this reset's sampled goals, not stale _tar_pos.
            goals=self._push_random_goals[env,zone] if self._random_at_goal else self._tar_pos[env,agent]
            targets=push_rsi_hand_targets(self._box_states[env,zone+self._push_slot_start][:,None],
                goals[:,None],self._box_size[env,zone+self._push_slot_start][:,None],self._relation_cfg)[:,0]
            body,body_rot,body_vel,body_ang,q=align_push_pose(body,body_rot,body_vel,body_ang,
                targets,self._key_body_ids[:2].cpu().tolist(),self._joint_rsi_spec['stand_off'])
        if skill=='push' and self._joint_rsi_spec.get('pose_mode')=='native_phase':
            from utils.native_push_rsi import safe_place
            body=safe_place(body,body_rot,self._box_states[env,zone+self._push_slot_start],
                self._box_size[env,zone+self._push_slot_start],targets,self._joint_rsi_bank.geometry)
        self._set_env_state(env,agent,body[:,0],multiply(q,rot),dof,rotate(q,vel),rotate(q,ang),dof_vel)
        self._kinematic_humanoid_rigid_body_states[env,agent]=torch.cat((body,body_rot,body_vel,body_ang),-1)
        self._every_env_init_dof_pos[env,agent]=dof
        self._reset_ref_slots[skill]=(env,agent)
        self._reset_ref_motion_ids[skill]=mids;self._reset_ref_motion_times[skill]=times
        self._rsi_interaction[env,agent]=True
        self._rsi_late[env,agent]=sample.get('late_requested',torch.zeros_like(times,dtype=torch.bool))
        self._rsi_phase[env,agent]=sample['phase']
        if skill=='push':
            self._push_episode_start[env,agent]=self._box_states[env,zone+self._push_slot_start,:3]
            distance=(body[:,self._key_body_ids[:2]]-targets).norm(dim=-1)
            for name,value in (('right_reset_distance',distance[:,0]),('left_reset_distance',distance[:,1])):
                key='push/rsi/'+name+'_sum'
                self._diag[key]=self._diag.get(key,0)+value.sum().detach()
            key='push/rsi/retarget_count'
            self._diag[key]=self._diag.get(key,0)+len(env)

    def _init_amp_obs_ref(self,slots,motion_ids,motion_times,skill_name):
        if skill_name!='push' or not self._joint_rsi_spec or self._joint_rsi_spec.get('pose_mode')=='native_phase':
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
        if self._push_scene_spec:
            heading=2*torch.pi*torch.rand(len(slot_env),device=self.device,generator=self._scene_generator)
        else:heading=root.new_full((len(slot_env),),torch.pi/2)
        yaw=heading-yaw_from_quat(root[:,3:7]);q=torch.zeros_like(root[:,3:7]);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
        lo,hi=self.cfg['env']['doorSpawnDistanceRange'];distance=lo+(hi-lo)*torch.rand(len(slot_env),device=self.device)
        push=self.relation_runtime.tasks[slot_env,slot_agent]==1
        target=root[:,:3].clone()
        target[:,0]=torch.where(push,-1.5,1.78)
        target[:,1]=(self._agent_zone[slot_env,slot_agent]-.5)*5.-distance-torch.where(push,self.cfg['env']['pushBox']['size'][1]/2,.1)
        if self._push_scene_spec:
            target[:,:2]=self._agent_spawn_offsets[self._agent_zone[slot_env,slot_agent],:2]
        expanded=q[:,None].expand_as(body[...,3:7])
        body[...,:3]=rotate(expanded,body[...,:3]-root[:,None,:3])+target[:,None]
        body[...,3:7]=multiply(expanded,body[...,3:7])
        body[...,7:10]=rotate(expanded,body[...,7:10]);body[...,10:13]=rotate(expanded,body[...,10:13])
        self._set_env_state(slot_env,slot_agent,target,multiply(q,root[:,3:7]),self._dof_pos[slot_env,slot_agent].clone(),
            rotate(q,root[:,7:10]),rotate(q,root[:,10:13]),self._dof_vel[slot_env,slot_agent].clone())
        self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent]=body

    def _sync_door_reset_states(self,ids):
        if self._push_only:return
        angles=self._door_dof_state[ids,:,0];panel=self._box_states[ids,:2].clone()
        q=torch.zeros_like(panel[...,3:7]);q[...,2]=torch.sin(angles/2);q[...,3]=torch.cos(angles/2)
        panel[...,3:7]=q;panel[...,7:]=0.;self._door_panel_states[ids]=panel
        handle=panel.clone();offset=panel.new_tensor([.78,-.1,1.]).expand(len(ids),2,3)
        handle[...,:3]+=rotate(q,offset);self._door_handle_states[ids]=handle

    def _reset_boxes(self,env_ids):
        self._box_states[env_ids]=self._initial_box_states[env_ids]
        self._box_states[env_ids,:,7:]=0.
        if not self._push_only:
            lo,hi=self._door_cfg['resetAngleRange'];self._door_dof_state[env_ids,:,0]=lo+(hi-lo)*torch.rand(len(env_ids),2,device=self.device)
            self._door_dof_state[env_ids,:,1]=0.
            self._sync_door_reset_states(env_ids)
        if self._push_scene_spec:
            from utils.push_scene import sample_layout
            if self._random_at_goal:
                from utils.push_scene import sample_random_at_scene
                xy,q,goals=sample_random_at_scene(len(env_ids),self._viewer_geometry_env,self.device,self._scene_generator)
                self._push_random_goals[env_ids,:,:2]=goals
                self._push_random_goals[env_ids,:,2]=self._box_states[env_ids,self._push_slot_start:,2]
            else:
                xy,q=sample_layout(len(env_ids),self._viewer_geometry_env,self.device,self._scene_generator)
            self._box_states[env_ids,self._push_slot_start:,:2]=xy
            self._box_states[env_ids,self._push_slot_start:,3:7]=q
            self._push_reset_states[env_ids]=self._box_states[env_ids,self._push_slot_start:]

    def _reset_task(self,env_ids):
        if self.cfg['env']['pushBox'].get('size_profiles'):
            self.relation_runtime.size=self._push_sizes()
        target=self.relation_runtime.start[env_ids].clone()
        if self._random_at_goal:
            target=self._zone_values(self._push_random_goals,env_ids)
        elif self._push_scene_spec:
            target+=self._push_goal_direction[env_ids]*self._relation_cfg['push']['target_distance']
        else:target[...,1]+=self._relation_cfg['push']['target_distance']
        self._tar_pos[env_ids]=torch.where((self.relation_runtime.tasks[env_ids]==1)[...,None],target,self.door_handle_position[env_ids])

    def _reset_env_tensors(self,env_ids):
        roots=[self._humanoid_actor_ids[env_ids].flatten(),self._box_actor_ids[env_ids].flatten()]
        if self._enable_markers:
            self._marker_pos[env_ids]=self._tar_pos[env_ids];roots.append(self._marker_actor_ids[env_ids].flatten())
        self._joint_reset_root_ids=torch.cat(roots).contiguous()
        self._joint_reset_dof_ids=self._humanoid_actor_ids[env_ids].flatten().contiguous() if self._push_only else torch.cat((self._humanoid_actor_ids[env_ids].flatten(),self._box_actor_ids[env_ids,:2].flatten())).contiguous()
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
        tasks=self._sample_tasks(env_ids)
        self.relation_runtime.tasks[env_ids]=tasks
        self._sample_box_assignments(env_ids)
        if self._push_scene_spec:
            if (tasks!=1).any():raise ValueError('Random PUSH layout cannot initialize OPEN tasks')
            from utils.push_scene import forward_xy
            if self._random_at_goal:
                delta=self._zone_values(self._push_random_goals,env_ids)-self._owned_push_states(env_ids)[...,:3]
                self._push_goal_distance[env_ids]=delta.norm(dim=-1)
                self._push_goal_direction[env_ids]=delta/self._push_goal_distance[env_ids,...,None].clamp_min(1e-6)
            else:
                self._push_goal_direction[env_ids,:,:2]=forward_xy(self._owned_push_states(env_ids)[...,3:7])
            self._push_goal_direction[env_ids,:,2]=0.
        self.relation_runtime.reset(env_ids,tasks,self._owned_push_states(env_ids))
        self._reset_actors(env_ids);self._reset_task(env_ids)
        if hasattr(self,'_push_episode_start'):
            self._push_episode_start[env_ids]=self._owned_push_states(env_ids)[...,:3]
        if self._manipulation_spec is not None:
            self.relation_runtime.initialize_manipulation(env_ids,self._owned_push_states(env_ids),
                self._tar_pos[env_ids],self.door_hinge_angle[env_ids])
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
            for o,handle in enumerate(self._box_handles[eid*self.num_objects:(eid+1)*self.num_objects]):
                owners=(self._agent_box_assignment[eid]==o).nonzero().flatten()
                active=bool(len(owners));owner=int(owners[0]) if active else 0
                color=self._agent_color(owner) if active else gymapi.Vec3(.45,.45,.45)
                for b in range(self._door_body_count if o<2 and not self._push_only else 1):
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
        if self._manipulation_spec is not None:
            self._body_hits.zero_();self._body_force_peak.zero_()
            self._clean_open_hits.zero_();self._clean_push_hits.zero_()
        if not self._pd_control:self._closing_forces[:,:2*self.num_dof]=(self.actions*self.motor_efforts.unsqueeze(0)*self.power_scale).reshape(self.num_envs,-1)
        if not self._push_only:self._closing_forces[:,-2:]=closing_torque(self._door_dof_state[...,0],self._door_dof_state[...,1],self._door_cfg['autoClose'])
        self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self._closing_forces))

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            collect_foot_support(self)
            hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]];forces=self._contact_forces[:,:,self._key_body_ids[:2]]
            open_hit=torch.zeros_like(self._support_hits)
            if not self._push_only:
                leaf=self._zone_values(self._door_forces)[:,:,[self._door_leaf_index,self._door_handle_index]].sum(-2)
                open_hit=hand_support_contact(hands,self.door_handle_position,forces,leaf,self.door_hinge_angle,self._hold_spec)
                self._support_hits|=open_hit
            from env.tasks.multi_agent.open_push_reward import push_hand_contact
            push_hit=push_hand_contact(hands,forces,self._owned_push_states(),self._zone_values(self._box_forces),self._push_sizes(),self._relation_cfg['push'],self._tar_pos,self._relation_cfg)
            self._push_hits|=push_hit
            if self._manipulation_spec is not None:
                from utils.manipulation_contact import nonhand_contact
                points=self._rigid_body_pos[:,:,self._nonhand_ids]+rotate(
                    self._rigid_body_rot[:,:,self._nonhand_ids],
                    self._nonhand_offsets.expand(self.num_envs,2,-1,-1))
                body_forces=self._contact_forces[:,:,self._nonhand_ids]
                open_body=handle_body=torch.zeros_like(self._support_hits)
                open_force=handle_force=torch.zeros_like(self.relation_runtime.elapsed)
                if not self._push_only:
                    panel=self._zone_values(self._door_panel_states).clone()
                    panel[...,:3]+=rotate(panel[...,3:7],panel.new_tensor([.45,0.,1.02]).expand(self.num_envs,2,3))
                    open_body,open_force=nonhand_contact(points,self._nonhand_radii,body_forces,
                        panel,[.9,.06,1.96],leaf,self._manipulation_spec)
                    handle_body,handle_force=nonhand_contact(points,self._nonhand_radii,body_forces,
                        self._zone_values(self._door_handle_states),[.14,.08,.04],leaf,self._manipulation_spec)
                push_body,push_force=nonhand_contact(points,self._nonhand_radii,body_forces,
                    self._owned_push_states(),self._push_sizes(),self._zone_values(self._box_forces),self._manipulation_spec)
                runtime=self.relation_runtime
                body=torch.where(runtime.tasks==0,open_body|handle_body,push_body)
                peak=torch.where(runtime.tasks==0,torch.maximum(open_force,handle_force),push_force)
                self._body_hits|=body
                self._body_force_peak.copy_(torch.maximum(self._body_force_peak,peak))
                posture=current_posture(self)
                push=self._owned_push_states();spec=self._relation_cfg['push']
                grounded=((push[...,2]-runtime.start[...,2]).abs()<=spec['ground_tolerance']) & (
                    1-2*(push[...,3].square()+push[...,4].square())>=torch.cos(push.new_tensor(spec['max_tilt'])))
                clean_open=open_hit & posture & ~body & (runtime.tasks==0)
                push_body_allowed=self._relation_cfg['push'].get('control_mode')=='midpoint_xy_net_credit'
                clean_push=push_hit & grounded & (runtime.tasks==1)
                if not push_body_allowed:clean_push &= ~body
                self._clean_open_hits|=clean_open;self._clean_push_hits|=clean_push
                runtime.collect_manipulation(push,self._tar_pos,self.door_hinge_angle,clean_open,clean_push)

    def _compute_reward(self,actions):
        runtime=self.relation_runtime;runtime.posture.copy_(current_posture(self));hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]]
        support_hits=self._support_hits
        if self._manipulation_spec is not None:
            runtime.body_contact.copy_(self._body_hits)
            runtime.clean_open_contact.copy_(self._clean_open_hits & ~self._body_hits)
            runtime.clean_push_contact.copy_(self._clean_push_hits)
            if self._relation_cfg['push'].get('control_mode')!='midpoint_xy_net_credit':
                runtime.clean_push_contact &= ~self._body_hits
            support_hits=self._clean_open_hits & ~self._body_hits
        near=(hands-self.door_handle_position[:,:,None]).norm(dim=-1).amin(-1)<=self._hold_spec['hand_distance']
        runtime.contact_age[:],runtime.contact[:]=support_with_grace(runtime.contact_age,support_hits & (runtime.tasks==0),near,self.dt,self._hold_spec['contact_grace'])
        if self._manipulation_spec is not None:
            runtime.contact_age.masked_fill_(runtime.body_contact,self._hold_spec['contact_grace']+self.dt)
            runtime.contact &= ~runtime.body_contact
        runtime.push_contact.copy_(self._push_hits & (runtime.tasks==1))
        # OPEN target follows its handle; PUSH target remains fixed throughout the episode.
        self._tar_pos.copy_(torch.where((runtime.tasks==0)[...,None],self.door_handle_position,self._tar_pos))
        r=runtime.step(self._humanoid_root_states[...,:3],hands,self.door_handle_position,self.door_hinge_angle,self._owned_push_states(),self._tar_pos)
        power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(r['reward'])
        collision=-self._agent_collision_coeff*compute_agent_collision_penalty(self._humanoid_root_states[...,:3],self._agent_collision_dist) if self._agent_collision_penalty else torch.zeros_like(power)
        total=r['reward']+power+collision;self.rew_buf[:]=total.flatten()
        terms=torch.stack((r['state'],r['progress'],r['bonus'],power,collision,total),-1).reshape(-1,len(self.REWARD_TERM_NAMES))
        self.extras['reward_terms']=terms;self._reward_term_sums+=terms.sum(0);self._reward_term_count+=len(terms)
        self.extras['open_push_tasks']=runtime.tasks.clone()
        if 'edge_state' in r:
            for index,name in enumerate(('push','at')):
                for key,value in (('state',r['edge_state']),('progress',r['edge_progress']),
                                  ('success_bonus',r['edge_bonus']),('own_success',runtime.own_success.float())):
                    metric='push_at/'+name+'/'+key+'_sum'
                    self._diag[metric]=self._diag.get(metric,0)+value[:,index::2].sum().detach()
        approach = approach_xy_diagnostics(self._humanoid_root_states[..., :3],
            self.door_handle_position, self._owned_push_states(), self._relation_cfg)
        for family,name in enumerate(('open','push')):
            mask=runtime.tasks==family;count=mask.sum().clamp_min(1)
            self._diag[name+'/samples']=self._diag.get(name+'/samples',0)+mask.sum().detach()
            self._diag[name+'/success_sum']=self._diag.get(name+'/success_sum',0)+(runtime.done & mask).sum().detach()
            metrics = dict(approach[name], manipulation_progress=r['progress'] /
                self._relation_cfg['progress_reward_weight'])
            for key, value in metrics.items():
                total_key = name + '/' + key + '_sum'
                self._diag[total_key] = self._diag.get(total_key, 0) + (value * mask).sum().detach()
        from env.tasks.multi_agent.open_push_reward import push_debug_metrics
        push_metrics=push_debug_metrics(hands,self._contact_forces[:,:,self._key_body_ids[:2]],
            self._owned_push_states(),self._zone_values(self._box_forces),runtime.start,self._push_sizes(),
            self._relation_cfg['push'],runtime.push_contact,runtime.push_eligible,self._push_episode_start,
            self._push_goal_direction if self._push_scene_spec else None,self._tar_pos,self._relation_cfg)
        self.extras['open_push_paired_contact']=runtime.push_contact.clone()
        self.extras['open_push_both_hands_near']=push_metrics['both_hands_near'].bool()
        mask=runtime.tasks==1
        for key,value in push_metrics.items():
            total_key='push/physics/'+key+'_sum'
            self._diag[total_key]=self._diag.get(total_key,0)+(value*mask).sum().detach()
        if self._manipulation_spec is not None:
            for family,name in enumerate(('open','push')):
                active=runtime.tasks==family
                raw=self._support_hits if family==0 else self._push_hits
                clean=runtime.clean_open_contact if family==0 else runtime.clean_push_contact
                moving=(self.door_hinge_velocity.abs()>.02) if family==0 else (self._owned_push_states()[...,7:10].norm(dim=-1)>.02)
                credited=runtime.open_credited/self._relation_cfg['door']['target_angle'] if family==0 else runtime.push_credited/self._relation_cfg['push']['target_distance']
                values=dict(hand_contact=raw.float(),clean_hand_contact=clean.float(),
                    body_contact=runtime.body_contact.float(),body_horizontal_force_N=self._body_force_peak,
                    credited_progress=credited)
                for phase,selected in (('all',active),('moving',active & moving & ~runtime.done),
                                       ('completed',active & runtime.done)):
                    prefix=name+'/contact/'+phase+'/'
                    self._diag[prefix+'count']=self._diag.get(prefix+'count',0)+selected.sum().detach()
                    for key,value in values.items():
                        self._diag[prefix+key+'_sum']=self._diag.get(prefix+key+'_sum',0)+(value*selected).sum().detach()
            key='push/contact/hand_credited_distance_sum'
            self._diag[key]=self._diag.get(key,0)+(runtime.push_hand_distance*(runtime.tasks==1)).sum().detach()
        if self._push_scene_spec:
            forward=((self._owned_push_states()[...,:3]-self._push_episode_start)*self._push_goal_direction).sum(-1)
        else:forward=self._owned_push_states()[...,1]-self._push_episode_start[...,1]
        for agent in range(2):
            for zone,name in enumerate(('lower','upper')):
                selected=(runtime.tasks[:,agent]==1) & (self._agent_zone[:,agent]==zone)
                prefix='push/zones/agent%d_%s/' % (agent,name)
                self._diag[prefix+'count']=self._diag.get(prefix+'count',0)+selected.sum().detach()
                for key,value in (('current_success',runtime.done[:,agent].float()),
                                  ('since_reset_forward_m',forward[:,agent]),
                                  ('hand_credited_m',runtime.push_hand_distance[:,agent])):
                    self._diag[prefix+key+'_sum']=self._diag.get(prefix+key+'_sum',0)+(value*selected).sum().detach()
        self.extras['open_push_zones']=self._agent_zone.clone()
        self._diag_steps+=1

    def _finish_relation_diagnostics(self):pass
    def consume_relation_diagnostics(self):
        result={}
        for name in ('open','push'):
            samples=self._diag.get(name+'/samples',torch.tensor(0.,device=self.device))
            result[name+'/current_success']=self._diag.get(name+'/success_sum',0)/samples.clamp_min(1)
            result[name+'/sample_count']=samples
            for key in ('distance_xy', 'xy_progress', 'manipulation_progress'):
                result[name+'/'+key] = self._diag.get(name+'/'+key+'_sum', 0) / samples.clamp_min(1)
            if self._joint_rsi_spec:
                for key in ('reset_count','interaction_count','late_request_count'):
                    result[name+'/rsi/'+key]=self._diag.get(name+'/rsi/'+key,torch.tensor(0.,device=self.device))
        samples=self._diag.get('push/samples',torch.tensor(0.,device=self.device))
        for key,value in self._diag.items():
            if key.startswith('push_at/') and key.endswith('_sum'):
                result[key[:-4]]=value/samples.clamp_min(1)
            if key.startswith('push/zones/'):
                prefix=key.rsplit('/',1)[0]+'/'
                if key.endswith('/count'):result[prefix+'sample_count']=value
                elif key.endswith('_sum'):result[key[:-4]]=value/self._diag[prefix+'count'].clamp_min(1)
            if key.startswith('push/physics/') and key.endswith('_sum'):
                result[key[:-4]]=value/samples.clamp_min(1)
            if '/contact/' in key and key.endswith('/count'):
                result[key[:-5]+'sample_count']=value
            elif '/contact/' in key and key.endswith('_sum'):
                prefix=key.rsplit('/',1)[0]+'/'
                denom=self._diag.get(prefix+'count',samples)
                result[key[:-4]]=value/denom.clamp_min(1)
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
