"""Independent Stage-1 PUSH and OPEN+HOLD with a unified scene Transformer."""
from utils.task_rsi import sample_reference, align_task_reference
import math
from dataclasses import replace
from isaacgym import gymapi, gymtorch
from utils import torch_utils
import torch
import torch.nn.functional as F
from utils.torch_utils import quat_mul, quat_rotate
from utils.motion_lib import MotionLib
from utils.push_door_spec import (phase_update, progress_reward, advance_success,
                                 expert_time, interaction_metadata, hand_handle_contact, validate_interaction, sample_start_layout, push_box_start_x, door_motion_matches)
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations, compute_agent_collision_penalty
from env.tasks.multi_agent.door_scene import DoorFixture
from tokenhsi.utils.door_asset import DoorSpec


class HumanoidMAPushDoor(HumanoidMA):
    REWARD_TERM_NAMES = ('push_progress', 'push_settled', 'door_progress', 'door_hold', 'success_bonus', 'total')

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_objects = cfg['env']['numObjects']
        self._extra_dofs_per_env = cfg['env']['numAgents']
        if self.num_objects != 2 * self._extra_dofs_per_env or self._extra_dofs_per_env != 2:
            raise ValueError('Push/door currently requires 2 agents and 4 objects (2 boxes + 2 doors)')
        self._interaction = cfg['env']['interaction']
        validate_interaction(self._interaction)
        self._relation_cfg = cfg['env']['relationReward']
        self._relation_graph_spec = cfg['env']['relationGraph']
        self._state_relation = self._edge_context = self._stage2 = False
        self._amp_task_conditioning = True
        self._enable_task_obs = True
        self._mode = 'test' if cfg['args'].test or cfg['args'].eval else 'train'
        self._task_preset = getattr(cfg['args'], 'task_graph', '') or 'random'
        if self._task_preset not in ('random', 'push', 'door', 'push_door', 'door_push'):
            raise ValueError('TASK_GRAPH must be random/push/door/push_door/door_push')
        if self._mode == 'train' and self._task_preset != 'random':
            raise ValueError('Training must use random independent task sampling')
        self._door_spec = replace(DoorSpec(), **self._interaction['door']['spring'])
        self._num_amp_obs_steps = cfg['env']['numAMPObsSteps']
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        N, M = self.num_envs, self.num_agents
        self._skill = ['loco', 'push', 'doorOpen']
        self._motion_lib = {skill: MotionLib(cfg['env']['motion_file'], skill,
            self._dof_body_ids, self._dof_offsets, self._key_body_ids.cpu().numpy(), self.device,
            motion_filter=(lambda path: door_motion_matches(path,self._interaction['door'].get('motion_direction','all'))) if skill=='doorOpen' else None)
            for skill in self._skill}
        self._num_amp_motion_features = 13 + self._dof_obs_size + self.num_dof + 3 * len(self._key_body_ids)
        self._num_amp_obs_per_step = self._num_amp_motion_features + 3
        self._amp_obs_buf = torch.zeros(N, M, self._num_amp_obs_steps, self._num_amp_obs_per_step, device=self.device)
        self._tasks = torch.zeros(N, M, dtype=torch.long, device=self.device)
        self._holding_phase = torch.zeros(N, M, dtype=torch.bool, device=self.device)
        self._elapsed = torch.zeros(N, M, device=self.device)
        self._done_task = torch.zeros_like(self._holding_phase)
        self._best_push = torch.zeros(N, M, device=self.device)
        self._best_angle = torch.zeros(N, M, device=self.device)
        self._targets = torch.zeros(N, M, 3, device=self.device)
        self._kinematic = self._initial_humanoid_rigid_body_states.clone()
        self._reward_term_sums = torch.zeros(len(self.REWARD_TERM_NAMES), device=self.device)
        self._reward_term_count = 0
        self._diagnostics = {}
        self.doors.bind_tensors()
        self._box_states = self._root_states.view(N, self.get_num_actors_per_env(), 13)[:, M:2*M]
        self._all_contacts = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self._box_actor_ids = self._humanoid_actor_ids + M
        self._door_state_view = self._dof_state.view(N, self._dofs_per_env, 2)[:, M*self.num_dof:]
        self._interaction_checkpoint_metadata = interaction_metadata(cfg['env'])
        self._validate_expert_windows()

    def get_task_obs_size(self): return self.num_objects*34 + self.num_agents*9
    def get_object_obs_size(self): return 34
    def get_goal_obs_size(self): return 9
    def get_scene_entity_sizes(self): return [self.get_clean_humanoid_obs_size(), 34, 9]
    def get_scene_normalized_entity_sizes(self): return [self.get_clean_humanoid_self_obs_size(), 30, 0]
    def get_scene_kinematic_size(self): return 7
    def get_scene_arena_scale(self): return self._arena_scale
    def get_relation_suffix_size(self): return 0
    def get_obs_size(self):
        return self.num_agents * (self.get_clean_humanoid_obs_size() + 9) + self.num_objects * 34 + (2*self.num_agents+self.num_objects)*7
    def get_num_amp_obs(self): return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.doors = DoorFixture(self.gym, self.sim, self._door_spec)
        box = self._interaction['push']['box']
        options = gymapi.AssetOptions(); options.density = box['density']
        self._box_size = torch.tensor(box['size'], device=self.device)
        self._box_asset = self.gym.create_box(self.sim, *box['size'], options)
        self._box_handles = []
        super()._create_envs(num_envs, spacing, num_per_row)
        # Isaac Gym state tensors use ENV-local coordinates; get_env_origin is
        # the viewer/world grid offset and must not be added to actor resets.
        self._env_origins.zero_()
        corners = torch.tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)], device=self.device)
        self._box_bps = corners * self._box_size / 2
        self._door_bps = corners * torch.tensor([self._door_spec.thickness, self._door_spec.width, self._door_spec.height], device=self.device)/2

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        handles = []
        for owner in range(self.num_agents):
            pose = gymapi.Transform(); pose.p = gymapi.Vec3(push_box_start_x(self._box_size,self._interaction['push']['target_distance'][1]), self._lane(owner), float(self._box_size[2]/2))
            actor = self.gym.create_actor(env_ptr, self._box_asset, pose, 'push_box_{}'.format(owner), env_id, 0)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor)
            for prop in props: prop.friction = self._interaction['push']['box']['friction']; prop.restitution=0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, actor, props)
            self.gym.set_rigid_body_color(env_ptr, actor, 0, gymapi.MESH_VISUAL, self._agent_color(owner))
            handles.append(actor)
        self._box_handles.append(handles)
        for owner in range(self.num_agents):
            pose = gymapi.Transform(); pose.p = gymapi.Vec3(0., self._lane(owner), 0.)
            self.doors.add(env_ptr, env_id, pose, 'door_{}'.format(owner))

    def _lane(self, owner): return (owner - (self.num_agents-1)/2) * self._interaction['lane_spacing']

    def _door_values(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        N, M = self.num_envs, self.num_agents
        values = self.doors.observe()
        return {key: value.view(N,M,-1)[ids] for key,value in values.items()}

    def _door_geometry(self, env_ids=None):
        """Compute geometry from reset DOF/base states too (no stale body pose)."""
        ids = slice(None) if env_ids is None else env_ids
        state = self._door_state_view[ids]
        base = self._root_states[self.doors.actor_indices.long()].view(self.num_envs,self.num_agents,13)[ids, :, :3]
        angle = state[...,0]
        rot = torch.zeros(*angle.shape,4,device=self.device)
        rot[...,2] = torch.sin(angle/2); rot[...,3] = torch.cos(angle/2)
        hinge = torch.tensor(self._door_spec.hinge,device=self.device,dtype=torch.float)
        local = torch.tensor(self._door_spec.front_handle_local,device=self.device,dtype=torch.float).expand(*angle.shape,3)
        handle = base + hinge + quat_rotate(rot.reshape(-1,4),local.reshape(-1,3)).view_as(local)
        back_local = local.clone(); back_local[...,0] *= -1
        back = base + hinge + quat_rotate(rot.reshape(-1,4),back_local.reshape(-1,3)).view_as(local)
        center_local = torch.tensor([0.,-self._door_spec.width/2,self._door_spec.height/2+self._door_spec.bottom_gap],device=self.device).expand_as(local)
        center = base+hinge+quat_rotate(rot.reshape(-1,4),center_local.reshape(-1,3)).view_as(local)
        return angle, state[...,1], center, rot, handle, back

    def _compute_observations(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        kin = None if env_ids is None else self._kinematic[env_ids]
        pos = self._rigid_body_pos if kin is None else kin[..., :3]
        rot = self._rigid_body_rot if kin is None else kin[...,3:7]
        vel = self._rigid_body_vel if kin is None else kin[...,7:10]
        angvel = self._rigid_body_ang_vel if kin is None else kin[...,10:13]
        B, M = pos.shape[:2]
        human = self._compute_clean_humanoid_nodes(pos,rot,vel,angvel,self._env_origins[ids])
        angle, hinge_vel, center, door_rot, handle, back = self._door_geometry(env_ids)
        boxes = self._box_states[ids]
        box_inv = boxes[...,3:7].clone(); box_inv[..., :3] *= -1
        bv = quat_rotate(box_inv.reshape(-1,4), boxes[...,7:10].reshape(-1,3)).view(B,M,3)
        bw = quat_rotate(box_inv.reshape(-1,4), boxes[...,10:13].reshape(-1,3)).view(B,M,3)
        box_nodes = torch.cat([bv,bw,self._box_bps.reshape(1,1,24).expand(B,M,-1),
            torch.ones(B,M,1,device=self.device),torch.zeros(B,M,3,device=self.device)],-1)
        door_vel = torch.zeros(B,M,6,device=self.device); door_vel[...,5]=hinge_vel
        door_nodes = torch.cat([door_vel,self._door_bps.reshape(1,1,24).expand(B,M,-1),
            torch.zeros(B,M,1,device=self.device),torch.ones(B,M,1,device=self.device),
            angle[...,None],hinge_vel[...,None]],-1)
        tasks = self._tasks[ids]
        duration = torch.where(tasks.bool(), self._interaction['door']['hold_seconds'],self._interaction['push']['settle_seconds'])
        goal = torch.cat([F.one_hot(tasks,2).float(),self._holding_phase[ids,...,None].float(),
            (self._elapsed[ids]/duration).clamp_max(1)[...,None],
            torch.full((B,M,1),math.radians(self._interaction['door']['open_degrees']),device=self.device),
            torch.full((B,M,1),self._interaction['push']['goal_tolerance'],device=self.device),self._done_task[ids,...,None].float(),self._best_push[ids,...,None],self._best_angle[ids,...,None]],-1)
        target_pos = torch.where(tasks[...,None].bool(),handle,self._targets[ids])
        human_heading = torch_utils.calc_heading_quat(rot[:,:,0].reshape(-1,4)).view(B,M,4)
        identity = torch.zeros(B,M,4,device=self.device); identity[...,3]=1
        poses = torch.cat([torch.cat([pos[:,:,0],human_heading],-1),torch.cat([boxes[...,:3],boxes[...,3:7]],-1),
            torch.cat([center,door_rot],-1),torch.cat([target_pos,identity],-1)],1)
        poses[...,:3] -= self._env_origins[ids,None,:]
        obs = torch.cat([human.flatten(1),box_nodes.flatten(1),door_nodes.flatten(1),goal.flatten(1),poses.flatten(1)],-1)
        if env_ids is None: self.obs_buf.copy_(obs)
        else: self.obs_buf[env_ids]=obs

    def _reset_envs(self, env_ids):
        if not len(env_ids): return
        E, M = len(env_ids), self.num_agents
        if self._task_preset == 'random':
            tasks = (torch.rand(E,M,device=self.device) < self._interaction['door_probability']).long()
        else:
            row={'push':[0,0],'door':[1,1],'push_door':[0,1],'door_push':[1,0]}[self._task_preset]
            tasks=torch.tensor(row,device=self.device).expand(E,M)
        self._tasks[env_ids]=tasks
        self._holding_phase[env_ids]=False; self._elapsed[env_ids]=0; self._done_task[env_ids]=False
        self._door_state_view[env_ids]=0
        boxes=self._box_states[env_ids].clone(); boxes.zero_(); boxes[...,6]=1
        lanes=torch.tensor([self._lane(a) for a in range(M)],device=self.device)
        origins=self._env_origins[env_ids,None,:]
        boxes[...,:3]=origins
        box_start_x=push_box_start_x(self._box_size,self._interaction['push']['target_distance'][1])
        boxes[...,0] += torch.where(tasks.bool(),-2.2,box_start_x)
        boxes[...,1] += lanes + tasks*.85
        boxes[...,2] += self._box_size[2]/2+.005
        self._box_states[env_ids]=boxes
        target=boxes[...,:3].clone()
        low,high=self._interaction['push']['target_distance']
        target[...,0]+=low+torch.rand(E,M,device=self.device)*(high-low)
        self._targets[env_ids]=target
        self._best_push[env_ids]=-torch.linalg.vector_norm(boxes[...,:2]-target[...,:2],dim=-1)
        self._best_angle[env_ids]=0
        spawn = self.cfg['env'].get('startRandomization')
        spawn_yaw = torch.zeros(E,M,device=self.device)
        if spawn is not None:
            boxes_xyz,target,human_xyz,spawn_yaw,box_yaw,shift = sample_start_layout(tasks,spawn,self._box_size,lanes)
            boxes[...,:3] = boxes_xyz + origins
            boxes[...,3:7] = 0
            boxes[...,5] = torch.sin(box_yaw/2);boxes[...,6] = torch.cos(box_yaw/2)
            self._box_states[env_ids] = boxes
            self._targets[env_ids] = target + origins
            self._best_push[env_ids] = -torch.linalg.vector_norm(boxes[...,:2]-self._targets[env_ids,...,:2],dim=-1)
            door_ids = self.doors.actor_indices.view(self.num_envs,M)[env_ids].long()
            door_root = self._root_states[door_ids].clone()
            door_root.zero_();door_root[...,6]=1
            door_root[...,:2] = shift + origins[...,:2]
            self._root_states[door_ids] = door_root
        if 'task_rsi' in self._interaction:
            root_pos,root_rot,dof_pos,body_pos,body_rot,rsi_mask=sample_reference(self,tasks)
        else:
            lib=self._motion_lib['loco']; mids=lib.sample_motions(E*M)
            times=torch.full((E*M,),self.dt*(self._num_amp_obs_steps-1),device=self.device)
            root_pos,root_rot,dof_pos,*_=lib.get_motion_state(mids,times)
            body_pos,body_rot,_,_=lib.get_motion_state_max(mids,times)
        unheading=torch_utils.calc_heading_quat_inv(root_rot)
        new_root=origins.expand(E,M,3).clone()
        push_gap=max(.65,float(torch.linalg.vector_norm(self._box_size[:2]/2))+.35)
        new_root[...,0]+=torch.where(tasks.bool(),-.65,box_start_x-push_gap)
        new_root[...,1]+=lanes-tasks*.36
        new_root[...,2]+=root_pos[:,2].view(E,M)
        if spawn is not None:
            new_root[...,:2] = human_xyz[...,:2] + origins[...,:2]
            heading = torch.zeros(E,M,4,device=self.device)
            heading[...,2]=torch.sin(spawn_yaw/2);heading[...,3]=torch.cos(spawn_yaw/2)
            unheading=quat_mul(heading.reshape(-1,4),unheading)
        if 'task_rsi' in self._interaction:
            unheading,new_root=align_task_reference(self,env_ids,tasks,rsi_mask,root_pos,body_pos,unheading,new_root)
        rotation=unheading[:,None,:].expand(-1,self.num_bodies,-1).reshape(-1,4)
        relative=body_pos-root_pos[:,None,:]
        transformed=quat_rotate(rotation,relative.reshape(-1,3)).view(E,M,self.num_bodies,3)+new_root[:,:,None,:]
        kin=torch.zeros(E,M,self.num_bodies,13,device=self.device)
        kin[...,:3]=transformed
        kin[...,3:7]=quat_mul(rotation,body_rot.reshape(-1,4)).view(E,M,self.num_bodies,4)
        self._kinematic[env_ids]=kin
        root=self._humanoid_root_states[env_ids].clone();root.zero_()
        root[...,:3]=new_root;root[...,3:7]=quat_mul(unheading,root_rot).view(E,M,4)
        self._humanoid_root_states[env_ids]=root
        self._dof_pos[env_ids]=dof_pos.view(E,M,-1);self._dof_vel[env_ids]=0
        actors=torch.cat([self._humanoid_actor_ids[env_ids].flatten(),self._box_actor_ids[env_ids].flatten(),
            self.doors.actor_indices.view(self.num_envs,M)[env_ids].flatten()]).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self._root_states),gymtorch.unwrap_tensor(actors),len(actors))
        articulated=torch.cat([self._humanoid_actor_ids[env_ids].flatten(),
            self.doors.actor_indices.view(self.num_envs,M)[env_ids].flatten()]).contiguous()
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self._dof_state),gymtorch.unwrap_tensor(articulated),len(articulated))
        if 'task_rsi' in self._interaction and self._pd_control:
            # Hold the reset pose until the policy supplies its first targets.
            # Otherwise a preview commit step drives joints toward stale/zero targets.
            reset_targets=self._dof_state[:,0].contiguous()
            humanoids=self._humanoid_actor_ids[env_ids].flatten().to(torch.int32).contiguous()
            self.gym.set_dof_position_target_tensor_indexed(self.sim,gymtorch.unwrap_tensor(reset_targets),gymtorch.unwrap_tensor(humanoids),len(humanoids))
        self.progress_buf[env_ids]=0;self.reset_buf[env_ids]=0;self._terminate_buf[self._flat_slot_ids(env_ids)]=0
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        self._compute_amp_observations(env_ids)
        self._amp_obs_buf[env_ids]=self._amp_obs_buf[env_ids,:,0:1].expand(-1,-1,self._num_amp_obs_steps,-1)

    def _compute_reward(self, actions):
        c=self._interaction; d=c['door']; p=c['push']
        angle,hinge_vel,_,_,handle,back=self._door_geometry()
        angle_target=math.radians(d['open_degrees'])
        self._holding_phase=phase_update(self._holding_phase,angle,angle_target,math.radians(d['reopen_degrees'])) & self._tasks.bool()
        distance=torch.linalg.vector_norm(self._box_states[...,:2]-self._targets[...,:2],dim=-1)
        door_progress,self._best_angle=progress_reward(angle.clamp(0,angle_target),self._best_angle,self.dt*d['progress_speed'])
        up=quat_rotate(self._box_states[...,3:7].reshape(-1,4),torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs*self.num_agents,3)).view(self.num_envs,self.num_agents,3)
        corners=self._box_bps[None,None,:,:].expand(self.num_envs,self.num_agents,-1,-1)
        rotations=self._box_states[...,3:7,None].transpose(-1,-2).expand(-1,-1,8,-1)
        rotated=quat_rotate(rotations.reshape(-1,4),corners.reshape(-1,3)).view(self.num_envs,self.num_agents,8,3)
        bottom=self._box_states[...,2]+rotated[...,2].min(-1).values
        grounded=bottom.abs()<p['ground_tolerance']
        upright=up[...,2]>.9
        push_progress,self._best_push=progress_reward(torch.where(grounded & upright,-distance,self._best_push),self._best_push,self.dt*p['progress_speed'])
        push_valid=(distance<p['goal_tolerance']) & (torch.linalg.vector_norm(self._box_states[...,7:10],dim=-1)<p['settle_speed']) & upright & grounded
        hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]]
        # Net forces alone cannot identify collision pairs. Require near-handle hands
        # and simultaneous forces on that same hand and handle link.
        hand_force=torch.linalg.vector_norm(self._contact_forces[:,:,self._key_body_ids[:2]],dim=-1)
        handle_pos=torch.stack([handle,back],dim=2)
        handle_force=torch.stack([torch.linalg.vector_norm(self._all_contacts[self.doors.body_indices[n]],dim=-1).view(self.num_envs,self.num_agents) for n in ('handle','handle_back')],-1)
        contact=hand_handle_contact(hands,handle_pos,hand_force,handle_force,d['contact_distance'],d['contact_force'])
        door_valid=(angle>=angle_target) & (hinge_vel.abs()<d['hold_max_speed']) & contact
        valid=torch.where(self._tasks.bool(),door_valid,push_valid)
        duration=torch.where(self._tasks.bool(),d['hold_seconds'],p['settle_seconds'])
        self._elapsed,self._done_task,first=advance_success(valid,self._elapsed,self._done_task,self.dt,duration)
        push=(self._tasks==0).float();door=1-push
        terms=torch.stack([c['progress_weight']*push_progress*push,c['maintain_weight']*push_valid*push,
            c['progress_weight']*door_progress*door,c['maintain_weight']*door_valid*door,
            c['success_weight']*first],-1)
        total=terms.sum(-1)
        if self.cfg['env'].get('agentCollisionPenalty', False):
            collision = -self.cfg['env'].get('agentCollisionCoeff', .5) * compute_agent_collision_penalty(
                self._humanoid_root_states[..., :3], self.cfg['env'].get('agentCollisionDist', .7))
            total = total + collision
            self.extras['agent_collision_penalty'] = collision.mean()
        self.rew_buf.copy_(total.flatten())
        terms=torch.cat([terms,total[...,None]],-1).flatten(0,1)
        self.extras['reward_terms']=terms
        self._reward_term_sums+=terms.mean(0);self._reward_term_count+=1
        self.extras['success']=self._done_task.flatten().float()
        self.extras['precision']=torch.where(self._tasks.bool(),(angle/angle_target).clamp(0,1),torch.exp(-distance)).flatten()
        self._diagnostics={'push_fraction':push.mean(),'door_hold_phase_fraction':self._holding_phase.float().mean(),
            'door_handle_contact_fraction':(contact*door).sum()/door.sum().clamp_min(1),
            'push_success':(self._done_task*push).sum()/push.sum().clamp_min(1),
            'door_success':(self._done_task*door).sum()/door.sum().clamp_min(1)}

    def _family(self, env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        return torch.where(self._tasks[ids]==0,0,torch.where(self._holding_phase[ids],2,1))

    def _compute_amp_observations(self, env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        kin=None if env_ids is None else self._kinematic[env_ids]
        pos=self._rigid_body_pos if kin is None else kin[...,:3]
        rot=self._rigid_body_rot if kin is None else kin[...,3:7]
        vel=self._rigid_body_vel if kin is None else kin[...,7:10]
        ang=self._rigid_body_ang_vel if kin is None else kin[...,10:13]
        B=pos.shape[0]*self.num_agents
        obs=build_amp_observations(pos[:,:,0].reshape(B,3),rot[:,:,0].reshape(B,4),vel[:,:,0].reshape(B,3),ang[:,:,0].reshape(B,3),
            self._dof_pos[ids].reshape(B,-1),self._dof_vel[ids].reshape(B,-1),pos[:,:,self._key_body_ids].reshape(B,-1,3),
            self._local_root_obs,self._root_height_obs,self._dof_obs_size,self._dof_offsets)
        family=self._family(env_ids)
        self._amp_obs_buf[ids,:,0]=torch.cat([obs,F.one_hot(family.flatten(),3).float()],-1).view(*family.shape,-1)
        # Whole physical history is conditioned on its CURRENT task/phase. Keeping
        # old labels would reveal transitions to the discriminator as a shortcut.
        self._amp_obs_buf[ids,:,:, -3:]=F.one_hot(family,3).float()[:,:,None,:].expand(-1,-1,self._num_amp_obs_steps,-1)

    def _expert_source(self, family):
        if family==0:return 'push',self._interaction['amp']['push_phase']
        if family==1:return 'doorOpen',self._interaction['amp']['open_phase']
        name='loco' if self._interaction['amp']['hold_source']=='loco' else 'doorOpen'
        return name,self._interaction['amp']['hold_phase'] if name=='doorOpen' else [0.,1.]

    def _validate_expert_windows(self):
        for family in range(3):
            name,phase=self._expert_source(family);lib=self._motion_lib[name]
            expert_time(lib._motion_lengths[lib._motion_weights>0],self.dt*(self._num_amp_obs_steps-1),phase)

    def fetch_amp_obs_demo(self, num_samples):
        # Populate every family even for small demo updates.
        family=torch.arange(num_samples,device=self.device)%3
        result=torch.empty(num_samples,self.get_num_amp_obs(),device=self.device)
        for uid in range(3):
            inds=(family==uid).nonzero(as_tuple=False).flatten()
            if not len(inds):continue
            name,phase=self._expert_source(uid);lib=self._motion_lib[name];mids=lib.sample_motions(len(inds))
            t=expert_time(lib.get_motion_length(mids),self.dt*(self._num_amp_obs_steps-1),phase)
            times=t[:,None]-self.dt*torch.arange(self._num_amp_obs_steps,device=self.device)
            motion_ids=mids[:,None].expand_as(times).reshape(-1)
            root,rot,dof,vel,ang,dofvel,key=lib.get_motion_state(motion_ids,times.flatten())
            obs=build_amp_observations(root,rot,vel,ang,dof,dofvel,key,self._local_root_obs,self._root_height_obs,self._dof_obs_size,self._dof_offsets)
            labels=F.one_hot(torch.full((len(obs),),uid,device=self.device),3).float()
            result[inds]=torch.cat([obs,labels],-1).view(len(inds),-1)
        return result

    def post_physics_step(self):
        self.progress_buf+=1;self._refresh_sim_tensors()
        self._compute_reward(self.actions);self._compute_observations();self._compute_reset()
        self._amp_obs_buf[:,:,1:]=self._amp_obs_buf[:,:,:-1].clone()
        self._compute_amp_observations()
        self.extras['amp_obs']=self._amp_obs_buf.flatten(0,1).flatten(1)
        self.extras['terminate']=self._terminate_buf
        self.extras['policy_obs']=self.obs_buf.clone()
        if self._recording:self._capture_video_frame()

    def consume_reward_term_means(self):
        if not self._reward_term_count:return None
        values=(self._reward_term_sums/self._reward_term_count).clone()
        self._reward_term_sums.zero_();self._reward_term_count=0
        return values

    def consume_relation_diagnostics(self):return self._diagnostics.copy()

    def _video_focus_points(self):
        return self._humanoid_root_states[0,:,:3]
