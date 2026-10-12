"""Stage-1 independent doors using the existing MA PPO/AMP and scene tokens."""
from pathlib import Path
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate
import torch
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, compute_agent_collision_penalty
from env.tasks.multi_agent.edge_door_reward import DoorRuntime
from env.tasks.multi_agent.edge_context_reward import owner_sum
from utils.edge_door_spec import compile_graph


class HumanoidMADoor(HumanoidMACarry):
    REWARD_TERM_NAMES = ('door_approach', 'door_interaction', 'door_progress',
        'door_success', 'power', 'collision', 'total')

    def __init__(self, cfg, *args, **kwargs):
        self._extra_object_dofs = cfg['env']['numObjects']
        self._door_cfg = cfg['env']['door']
        self._door_reference_spec = cfg['env'].get('doorReference')
        self._door_reference_bank = None
        self._door_approach_spec = cfg['env'].get('doorApproachAMP')
        self._door_approach_gate = None
        super().__init__(cfg, *args, **kwargs)
        if self._door_approach_spec:
            # Actor/body tensors for these Gym environments are env-local. Keep
            # GTA pose records in that same frame instead of subtracting grid XY.
            self._env_origins.zero_()
        if self._door_reference_spec:
            from utils.door_reference import DoorReferenceBank
            self._door_reference_bank = DoorReferenceBank(self._motion_lib,
                self._key_body_ids[:2].cpu().tolist(),self._key_body_ids[2:].cpu().tolist(),
                self._door_reference_spec,self.dt*(self._num_amp_obs_steps-1))
        if self._door_approach_spec:
            from utils.door_approach_amp import DoorApproachAMP
            self._door_approach_gate = DoorApproachAMP(self.num_envs,self.num_agents,self.device,self._door_approach_spec)
        self._amp_expert_skills = list(cfg['env']['skill'])
        self._skill_disc_prob.copy_(torch.tensor(cfg['env']['ampExpertMapping']['OPEN'],device=self.device))

    def get_door_experiment_contract(self):
        if not self._door_reference_spec and not self._door_approach_spec and not self._door_cfg.get('hinge'):
            return None  # original Door ON/OFF checkpoint contract remains valid
        import yaml
        import hashlib
        with open(self.cfg['args'].cfg_train) as file:
            train=yaml.safe_load(file)
        contract={'share_edge_encoder':train['params']['network']['transformer'].get('share_edge_encoder',False),
            'motion_manifest_sha256':hashlib.sha256(Path(self.cfg['env']['motion_file']).read_bytes()).hexdigest(),
            'reference':self._door_reference_spec,'expert_skills':self.cfg['env']['skill'],
            'expert_mapping':self.cfg['env']['ampExpertMapping'],'rsi':self.cfg['env']['skillInitProb']}
        if self._door_approach_spec:
            contract['approach_amp']=self._door_approach_spec
            contract['reset_revision']=3  # batched setters, env-local poses, stable clip starts
            contract['state_init']=self.cfg['env']['stateInit']
        if self._door_cfg.get('openingDirection', 'push') == 'pull':
            contract['opening_direction'] = 'pull'
            contract['handle_offset'] = self.door_local_handle_offset
        if self._door_cfg.get('hinge'):
            contract['hinge_physics'] = dict(self._door_cfg['hinge'])
        return contract

    def fetch_amp_obs_demo(self, num_samples):
        if not self.cfg['env'].get('ampEnabled', True):
            return torch.zeros(num_samples,self.get_num_amp_obs(),device=self.device)
        if not getattr(self,'_door_reference_bank',None):
            return super().fetch_amp_obs_demo(num_samples)
        selected=torch.multinomial(self._skill_disc_prob,num_samples,replacement=True)
        result=torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        for i,skill in enumerate(self._amp_expert_skills):
            rows=(selected==i).nonzero().flatten()
            if not len(rows):continue
            lib=self._motion_lib[skill]
            if skill in self._door_reference_bank.pools:
                ids,times=self._door_reference_bank.sample(skill,len(rows))
            else:
                ids=lib.sample_motions(len(rows))
                history=self.dt*(self._num_amp_obs_steps-1)
                times=lib.sample_time(ids,truncate_time=history)+history
            result[rows]=self.build_amp_obs_demo(ids,times,lib).reshape(len(rows),-1)
        return result

    def get_clean_object_obs_size(self):
        return 35  # existing O30 + hinge angle/velocity + local handle XYZ

    @property
    def door_local_handle_offset(self):
        # Pull handle has clearance for a rigid humanoid hand behind the grip.
        return (.78, -.25, 1.) if self._door_cfg.get('openingDirection', 'push') == 'pull' else (.78, -.10, 1.)

    def _build_agent_spawn_offsets(self, device_type, device_id):
        super()._build_agent_spawn_offsets(device_type, device_id)
        # Each owner starts 0.8m in front of its own handle; workspaces are 5m apart.
        self._agent_spawn_offsets[:] = self._agent_spawn_offsets.new_tensor([
            [1.78,-3.3,0.], [1.78,1.7,0.]])
        if self._door_approach_spec:
            distance=self._door_approach_spec['spawn_distance']
            self._agent_spawn_offsets[:,1]=self._agent_spawn_offsets.new_tensor([-2.5,2.5])+self.door_local_handle_offset[1]-distance

    def _load_box_asset(self):
        self._box_size = torch.tensor([.9,.06,2.],device=self.device).view(1,1,3).repeat(self.num_envs,self.num_objects,1)
        options=gymapi.AssetOptions()
        options.fix_base_link=True
        options.collapse_fixed_joints=False
        options.disable_gravity=False
        options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
        options.angular_damping=.05
        asset_path=Path(__file__).resolve().parents[3]/'assets'/'door'
        asset_file = getattr(self, '_door_asset_file',
            'door_pull.urdf' if self._door_cfg.get('openingDirection', 'push') == 'pull' else 'door.urdf')
        self._door_asset=self.gym.load_asset(self.sim,str(asset_path),asset_file,options)
        if self.gym.get_asset_dof_count(self._door_asset) != 1:
            raise ValueError('Door asset must have one revolute DOF')
        self._door_body_count=self.gym.get_asset_rigid_body_count(self._door_asset)
        self._door_leaf_index=self.gym.find_asset_rigid_body_index(self._door_asset,'panel')
        self._door_handle_index=self.gym.find_asset_rigid_body_index(self._door_asset,'handle')
        if min(self._door_leaf_index,self._door_handle_index)<0:
            raise ValueError('Door asset requires panel and handle links')

    def _build_box(self, env_id, object_id, env_ptr):
        pose=gymapi.Transform()
        pose.p=gymapi.Vec3(1., (object_id-.5)*5., 0.)
        handle=self.gym.create_actor(env_ptr,self._door_asset,pose,
            'door_{}'.format(object_id),env_id,0,0)
        props=self.gym.get_actor_dof_properties(env_ptr,handle)
        props['driveMode']=gymapi.DOF_MODE_NONE
        props['stiffness']=0.
        hinge = self._door_cfg.get('hinge')
        props['damping'] = hinge['damping'] if hinge else .5
        if hinge:
            props['friction'] = hinge['friction']
        self.gym.set_actor_dof_properties(env_ptr,handle,props)
        self._box_handles.append(handle)

    def _build_box_bps(self):
        super()._build_box_bps()
        self._box_bps += self._box_bps.new_tensor([.45,0.,1.])

    def _build_box_tensors(self):
        super()._build_box_tensors()
        self._door_dof_state=self._dof_state.view(self.num_envs,-1,2)[:,self.num_agents*self.num_dof:]
        self._door_bodies=self._rigid_body_state.view(self.num_envs,-1,13)[:,self.num_agents*self.num_bodies:self.num_agents*self.num_bodies+self.num_objects*self._door_body_count].view(self.num_envs,self.num_objects,self._door_body_count,13)
        self._door_panel_states=self._door_bodies[:,:,self._door_leaf_index]
        self._door_handle_states=self._door_bodies[:,:,self._door_handle_index]

    @property
    def door_hinge_angle(self):
        return self._door_dof_state[...,0]

    @property
    def door_hinge_velocity(self):
        return self._door_dof_state[...,1]

    @property
    def door_pose(self):
        return self._door_panel_states[...,:7]

    @property
    def door_handle_position(self):
        return self._door_handle_states[...,:3]

    def _logical_box_values(self, values, env_ids=None):
        if values is self._box_states:
            values=self._door_panel_states
        return values if env_ids is None else values[env_ids]

    def _clean_object_features(self, object_local, env_ids):
        ids=slice(None) if env_ids is None else env_ids
        panel=self._door_panel_states[ids]
        inv=panel[...,3:7].clone(); inv[...,:3]*=-1
        local_handle=quat_rotate(inv.reshape(-1,4),
            (self.door_handle_position[ids]-panel[...,:3]).reshape(-1,3)).view(*panel.shape[:-1],3)
        return torch.cat((object_local,self._door_dof_state[ids],local_handle),-1)

    def _init_relation_runtime(self):
        graph=compile_graph(self._relation_graph_spec,self.num_agents,self.num_objects,self.device)
        self.relation_runtime=DoorRuntime(self.num_envs,graph,self._relation_cfg,self.device)
        self._door_diag_sums={}
        self._door_diag_steps=0

    def _sample_box_assignments(self, env_ids):
        # Physical door slots and owner bindings remain independent of hinge reset.
        slots=torch.arange(self.num_objects,device=self.device)
        self._logical_box_order[env_ids]=slots
        self._agent_box_assignment[env_ids]=slots

    def reset_door_angles(self, env_ids, angles=None, apply_to_sim=True):
        """Reset any env subset; angles [B,2], without changing graph binding."""
        if angles is None:
            lo,hi=self._door_cfg['resetAngleRange']
            angles=lo+(hi-lo)*torch.rand(len(env_ids),self.num_objects,device=self.device)
        angles=torch.as_tensor(angles,device=self.device,dtype=self._door_dof_state.dtype)
        if angles.shape != (len(env_ids),self.num_objects) or not torch.isfinite(angles).all() or ((angles<0)|(angles>1.7)).any():
            raise ValueError('Door reset angles require finite [B,2] values within [0,1.7]')
        self._door_dof_state[env_ids,:,0]=angles
        self._door_dof_state[env_ids,:,1]=0.
        if apply_to_sim:
            self._door_angle_actor_ids = self._box_actor_ids[env_ids].flatten().contiguous()
            ids = self._door_angle_actor_ids
            self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self._dof_state),gymtorch.unwrap_tensor(ids),len(ids))
        self._sync_door_reset_states(env_ids)

    def _sync_door_reset_states(self, ids):
        # Gym rigid-body tensors can remain stale until simulation advances.
        # Seed exact forward kinematics for reset observations (hinge at frame origin).
        sign = -1 if self._door_cfg.get('openingDirection', 'push') == 'pull' else 1
        q=sign*self.door_hinge_angle[ids]
        frame=self._box_states[ids]
        rotation=torch.zeros_like(frame[...,3:7])
        rotation[...,2]=torch.sin(q/2);rotation[...,3]=torch.cos(q/2)
        panel=frame.clone();panel[...,3:7]=rotation;panel[...,7:]=0.
        self._door_panel_states[ids]=panel
        offset=torch.tensor(self.door_local_handle_offset,device=self.device).expand(len(ids),self.num_objects,3)
        handle=panel.clone();handle[...,:3]+=quat_rotate(rotation.reshape(-1,4),offset.reshape(-1,3)).view_as(offset)
        self._door_handle_states[ids]=handle

    def _reset_boxes(self, env_ids):
        self._box_states[env_ids]=self._initial_box_states[env_ids]
        self.reset_door_angles(env_ids, apply_to_sim=False)

    def _reset_ref_state_init(self, slot_env, slot_agent):
        if not self._door_reference_bank or self._is_eval:
            super()._reset_ref_state_init(slot_env,slot_agent)
            if self._door_approach_spec:
                # Gym actor states are env-local. Remove the inherited arena
                # origin and clip translation, and face the owner's door (+Y).
                from utils.door_reference import multiply, rotate, yaw_from_quat
                import math
                root = self._humanoid_root_states[slot_env, slot_agent].clone()
                body = self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent].clone()
                yaw = math.pi / 2 - yaw_from_quat(root[:, 3:7])
                q = torch.zeros_like(root[:, 3:7])
                q[:, 2] = torch.sin(yaw / 2)
                q[:, 3] = torch.cos(yaw / 2)
                target = root[:, :3].clone()
                target[:, :2] = self._agent_spawn_offsets[slot_agent, :2]
                expanded = q[:, None].expand_as(body[..., 3:7])
                body[..., :3] = rotate(expanded, body[..., :3] - root[:, None, :3]) + target[:, None]
                body[..., 3:7] = multiply(expanded, body[..., 3:7])
                body[..., 7:10] = rotate(expanded, body[..., 7:10])
                body[..., 10:13] = rotate(expanded, body[..., 10:13])
                self._set_env_state(slot_env, slot_agent, target,
                    multiply(q, root[:, 3:7]), self._dof_pos[slot_env, slot_agent].clone(),
                    rotate(q, root[:, 7:10]), rotate(q, root[:, 10:13]),
                    self._dof_vel[slot_env, slot_agent].clone())
                self._kinematic_humanoid_rigid_body_states[slot_env, slot_agent] = body
            return
        from utils.door_reference import align_reference, multiply, rotate, yaw_from_quat
        import math
        selected=torch.multinomial(self._skill_init_prob,len(slot_env),replacement=True)
        for i,skill in enumerate(self._skill):
            chosen=(selected==i).nonzero().flatten()
            if not len(chosen):continue
            env,agent=slot_env[chosen],slot_agent[chosen]
            lib=self._motion_lib[skill]
            if skill in self._door_reference_bank.pools:
                ids,times=self._door_reference_bank.sample(skill,len(chosen),late=True)
            else:
                ids=lib.sample_motions(len(chosen));times=lib.sample_time_rsi(ids)
            root,rot,dof,vel,ang,dof_vel,key=lib.get_motion_state(ids,times)
            body,body_rot,body_vel,body_ang=lib.get_motion_state_max(ids,times)
            if skill in self._door_reference_bank.pools:
                body,body_rot,body_vel,body_ang,q=align_reference(body,body_rot,body_vel,
                    body_ang,rot,self.door_handle_position[env,agent],
                    self._key_body_ids[:2].cpu().tolist(),self._door_reference_spec['stand_off'],self._door_reference_spec['min_forward_reach'])
            else:
                yaw=math.pi/2-yaw_from_quat(rot)
                q=torch.zeros_like(rot);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
                target=root.clone();target[:,:2]=(self._env_origins[env]+self._agent_spawn_offsets[agent])[:,:2]
                body=rotate(q[:,None].expand_as(body_rot),body-root[:,None])+target[:,None]
                body_rot=multiply(q[:,None].expand_as(body_rot),body_rot)
                body_vel=rotate(q[:,None].expand_as(body_rot),body_vel)
                body_ang=rotate(q[:,None].expand_as(body_rot),body_ang)
            self._set_env_state(env,agent,body[:,0],multiply(q,rot),dof,
                rotate(q,vel),rotate(q,ang),dof_vel)
            self._kinematic_humanoid_rigid_body_states[env,agent]=torch.cat((body,body_rot,body_vel,body_ang),-1)
            self._every_env_init_dof_pos[env,agent]=dof
            self._reset_ref_slots[skill]=(env,agent)
            self._reset_ref_motion_ids[skill]=ids
            self._reset_ref_motion_times[skill]=times

    def _reset_env_tensors(self, env_ids):
        # GPU PhysX retains one pending indexed setter per tensor. Submit humans
        # and doors together so the later object reset cannot replace human RSI.
        root_ids = [self._humanoid_actor_ids[env_ids].flatten(),
                    self._box_actor_ids[env_ids].flatten()]
        if self._enable_markers:
            self._marker_pos[env_ids] = self._tar_pos[env_ids]
            root_ids.append(self._marker_actor_ids[env_ids].flatten())
        self._door_reset_root_ids = torch.cat(root_ids).contiguous()
        root_ids = self._door_reset_root_ids
        self._door_reset_dof_ids = torch.cat((self._humanoid_actor_ids[env_ids].flatten(),
                             self._box_actor_ids[env_ids].flatten())).contiguous()
        dof_ids = self._door_reset_dof_ids
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(root_ids), len(root_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dof_ids), len(dof_ids))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        flat = self._flat_slot_ids(env_ids)
        self._terminate_buf[flat] = 0
        if self._is_eval:
            self._success_buf[flat] = 0
            self._precision_buf[flat] = float('Inf')

    def _reset_task(self, env_ids):
        self._tar_pos[env_ids]=self.door_handle_position[env_ids]

    def _reset_envs(self, env_ids):
        if not len(env_ids):
            return
        self._reset_default_slots=None
        self._reset_ref_slots={};self._reset_ref_motion_ids={};self._reset_ref_motion_times={}
        if self._door_reference_bank:
            self._reset_boxes(env_ids)
        self._reset_actors(env_ids)
        self._sample_box_assignments(env_ids)
        if not getattr(self,'_door_reference_bank',None):
            self._reset_boxes(env_ids)
        self._reset_task(env_ids)
        self._reset_env_tensors(env_ids)
        # Reset observations use reference kinematics. Refresh only after the
        # next simulation so staged root/DOF writes are not overwritten.
        self._sync_door_reset_states(env_ids)
        self.relation_runtime.reset(env_ids,self.door_hinge_angle[env_ids])
        if self._door_approach_gate is not None:
            self._door_approach_gate.reset(env_ids)
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)

    def _compute_reward(self, actions):
        if self._door_approach_gate is not None:
            weight=self._door_approach_gate.step(self._humanoid_root_states[...,:3],
                self._rigid_body_pos[:,:,self._key_body_ids[:2]],self.door_handle_position,self.door_hinge_angle)
            self.extras['door_amp_weight']=weight.reshape(-1,1).clone()
            for key,value in {'open/amp_weight':weight.mean(),
                    'open/amp_active_fraction':(weight>0).float().mean()}.items():
                self._door_diag_sums[key]=self._door_diag_sums.get(key,0)+value.detach()
        result=self.relation_runtime.step(self._humanoid_root_states[...,:3],
            self._rigid_body_pos[:,:,self._key_body_ids[:2]],self.door_handle_position,self.door_hinge_angle)
        power=torch.zeros_like(result['agent_task_reward'])
        collision=torch.zeros_like(power)
        if self._power_reward:
            power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1)
        if self._agent_collision_penalty:
            collision=-self._agent_collision_coeff*compute_agent_collision_penalty(self._humanoid_root_states[...,:3],self._agent_collision_dist)
        total=result['agent_task_reward']+power+collision
        self.rew_buf[:]=total.flatten()
        graph=self.relation_runtime.graph
        terms=[owner_sum(result[k],graph) for k in ('approach_component','interaction_component','progress_component','success_component')]
        terms=torch.stack(terms+[power,collision,total],-1).reshape(-1,len(self.REWARD_TERM_NAMES))
        self.extras['reward_terms']=terms
        self._reward_term_sums+=terms.sum(0);self._reward_term_count+=terms.shape[0]
        self._tar_pos.copy_(self.door_handle_position)
        for k,v in {'open/hinge_angle':self.door_hinge_angle.mean(),
                    'open/hinge_velocity':self.door_hinge_velocity.abs().mean(),
                    'open/current_agent_success':self.relation_runtime.done.float().mean(),
                    'open/current_scene_success':self.relation_runtime.done.all(-1).float().mean()}.items():
            self._door_diag_sums[k]=self._door_diag_sums.get(k,0)+v.detach()
        self._door_diag_steps+=1

    def _finish_relation_diagnostics(self):
        pass

    def consume_relation_diagnostics(self):
        result={k:v/max(1,self._door_diag_steps) for k,v in self._door_diag_sums.items()}
        self._door_diag_sums={};self._door_diag_steps=0
        return result

    def _compute_metrics_evaluation(self):
        self._success_buf.copy_(self.relation_runtime.done.flatten().long())
        error=(self._relation_cfg['door']['target_angle']-self.door_hinge_angle).clamp_min(0)
        self._precision_buf.copy_(error.flatten())
