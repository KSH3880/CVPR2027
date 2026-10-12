"""Single-person push/open/continuous-support task. No second-agent transport."""
from isaacgym import gymapi, gymtorch
import torch
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor
from env.tasks.multi_agent.edge_door_reward import DoorRuntime, door_reward
from env.tasks.multi_agent.edge_context_reward import owner_sum, goal_success
from utils.edge_door_spec import compile_graph
from utils.door_hold import validate_hold, closing_torque, hand_support_contact, update_hold, support_with_grace, init_posture_bodies, current_posture, collect_foot_support


class DoorHoldRuntime(DoorRuntime):
    def __init__(self, num_envs, graph, config, device, hold_spec, dt):
        super().__init__(num_envs,graph,config,device)
        self.spec,self.dt=hold_spec,dt
        self.elapsed=torch.zeros_like(self.phi)
        self.contact=torch.zeros_like(self.own_success)
        self.posture=torch.zeros_like(self.own_success)
        self.contact_age=torch.full_like(self.phi,hold_spec['contact_grace']+dt)

    def reset(self, ids, angles):
        super().reset(ids,angles)
        self.elapsed[ids]=0
        self.contact[ids]=False
        self.posture[ids]=False
        self.contact_age[ids]=self.spec['contact_grace']+self.dt
        self.own_success[ids]=False;self.done[ids]=False;self.achieved[ids]=False

    def step(self, roots, hands, handles, angles):
        d=self.config['door']
        self.elapsed[:],success=update_hold(self.elapsed,angles,self.contact & self.posture,
            d['target_angle']-d['angle_tolerance'],self.spec['duration'],self.dt)
        result=door_reward(roots,hands,handles,angles,self.graph,self.config,success_override=success)
        # Preserve approach shaping, reward reaching/contact before success,
        # and split progress between opening and elapsed supported hold.
        result['interaction_component']*=(.5+.5*self.contact.float())*self.posture.float()
        angle_progress=(angles/d['target_angle']).clamp(0,1)*self.posture.float()
        hold_progress=self.elapsed/self.spec['duration']
        result['progress_component']=torch.where(success,torch.ones_like(angles),
            .5*angle_progress+.5*hold_progress)*self.config['progress_reward_weight']
        result['total']=sum(result[k] for k in ('approach_component','interaction_component','progress_component','success_component'))
        result['agent_task_reward']=owner_sum(result['total'],self.graph)  # single owner
        result['phi_raw']=.5*angle_progress+.5*hold_progress
        result['hold_elapsed']=self.elapsed.clone();result['support_contact']=self.contact.clone()
        self.phi.copy_(result['phi_raw']);self.own_success.copy_(success)
        self.achieved|=success;self.done.copy_(goal_success(success,self.graph));self.last_result=result
        return result


class HumanoidMADoorHold(HumanoidMADoor):
    def __init__(self,cfg,*args,**kwargs):
        validate_hold(cfg['env'])
        self._hold_spec=cfg['env']['doorHold']
        self._door_asset_file='door_hold.urdf'
        super().__init__(cfg,*args,**kwargs)
        init_posture_bodies(self)
        self._closing_forces=torch.zeros(self.num_envs,self.num_agents*self.num_dof+self.num_objects,device=self.device)
        self._support_hits=torch.zeros(self.num_envs,self.num_objects,dtype=torch.bool,device=self.device)

    def _build_agent_spawn_offsets(self,device_type,device_id):
        HumanoidMA._build_agent_spawn_offsets(self,device_type,device_id)
        distance=self._door_approach_spec['spawn_distance']
        self._agent_spawn_offsets[:]=self._agent_spawn_offsets.new_tensor([[1.78,-2.5+self.door_local_handle_offset[1]-distance,0.]])

    def _build_box(self,env_id,object_id,env_ptr):
        super()._build_box(env_id,object_id,env_ptr)
        handle=self._box_handles[-1]
        props=self.gym.get_actor_dof_properties(env_ptr,handle)
        # The policy has no door action: only an environmental closing spring.
        props['driveMode']=gymapi.DOF_MODE_EFFORT
        props['stiffness']=0.
        self.gym.set_actor_dof_properties(env_ptr,handle,props)

    def _reset_ref_state_init(self,slot_env,slot_agent):
        super()._reset_ref_state_init(slot_env,slot_agent)
        lo,hi=self.cfg['env']['doorSpawnDistanceRange']
        distance=lo+(hi-lo)*torch.rand(len(slot_env),device=self.device)
        root=self._humanoid_root_states[slot_env,slot_agent].clone()
        delta=torch.zeros_like(root[:,:3])
        delta[:,1]=self._door_approach_spec['spawn_distance']-distance
        self._set_env_state(slot_env,slot_agent,root[:,:3]+delta,root[:,3:7],
            self._dof_pos[slot_env,slot_agent].clone(),root[:,7:10],root[:,10:13],
            self._dof_vel[slot_env,slot_agent].clone())
        body=self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent].clone()
        body[...,:3]+=delta[:,None]
        self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent]=body

    def _load_box_asset(self):
        super()._load_box_asset()
        self._box_size[...,2]=1.96

    def _build_box_bps(self):
        super()._build_box_bps()
        self._box_bps+=self._box_bps.new_tensor([0.,0.,.02])

    def _build_box_tensors(self):
        super()._build_box_tensors()
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs,-1,3)
        start=self.num_agents*self.num_bodies
        self._leaf_contact_forces=forces[:,start:start+self.num_objects*self._door_body_count].view(self.num_envs,self.num_objects,self._door_body_count,3)

    def _init_relation_runtime(self):
        graph=compile_graph(self._relation_graph_spec,self.num_agents,self.num_objects,self.device)
        self.relation_runtime=DoorHoldRuntime(self.num_envs,graph,self._relation_cfg,self.device,self._hold_spec,self.dt)
        self._door_diag_sums={};self._door_diag_steps=0

    def get_clean_object_obs_size(self):
        return 38  # O35 + supported time, contact flag and contact age

    def _clean_object_features(self,object_local,env_ids):
        values=super()._clean_object_features(object_local,env_ids)
        runtime=getattr(self,'relation_runtime',None)
        if runtime is None:
            extra=values.new_zeros(*values.shape[:-1],3)
        else:
            ids=slice(None) if env_ids is None else env_ids
            extra=torch.stack((runtime.elapsed[ids]/self._hold_spec['duration'],runtime.contact[ids].float(),
                runtime.contact_age[ids]/(self._hold_spec['contact_grace']+self.dt)),-1)
        return torch.cat((values,extra),-1)

    def pre_physics_step(self,actions):
        super().pre_physics_step(actions)
        self._posture_foot_forces.zero_()
        self._support_hits.zero_()
        self._closing_forces.zero_()
        if not self._pd_control:
            self._closing_forces[:,:self.num_agents*self.num_dof]=(self.actions*self.motor_efforts.unsqueeze(0)*self.power_scale).reshape(self.num_envs,-1)
        self._closing_forces[:,-self.num_objects:]=closing_torque(self.door_hinge_angle,self.door_hinge_velocity,self._door_cfg['autoClose'])
        self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self._closing_forces))

    def _physics_step(self):
        # Contact impulses can occur on the first physics tick and disappear
        # on the last. Collect each tick rather than missing real hand support.
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim,True)
            self._refresh_sim_tensors()
            collect_foot_support(self)
            hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]]
            forces=self._contact_forces[:,:,self._key_body_ids[:2]]
            leaf=self._leaf_contact_forces[:,:,[self._door_leaf_index,self._door_handle_index]].sum(-2)
            self._support_hits|=hand_support_contact(hands,self.door_handle_position,forces,leaf,self.door_hinge_angle,self._hold_spec)

    def _compute_reward(self,actions):
        hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]]
        near=(hands-self.door_handle_position[:,:,None]).norm(dim=-1).amin(-1)<=self._hold_spec['hand_distance']
        runtime=self.relation_runtime
        runtime.posture.copy_(current_posture(self))
        age,contact=support_with_grace(runtime.contact_age,self._support_hits,near,self.dt,self._hold_spec['contact_grace'])
        runtime.contact_age.copy_(age);runtime.contact.copy_(contact)
        super()._compute_reward(actions)
        self.extras['door_hold_elapsed']=self.relation_runtime.elapsed.clone()
        self.extras['door_support_contact']=self.relation_runtime.contact.clone()
        self._door_diag_sums['open/hold_seconds']=self._door_diag_sums.get('open/hold_seconds',0)+self.relation_runtime.elapsed.mean().detach()
        self._door_diag_sums['open/support_contact']=self._door_diag_sums.get('open/support_contact',0)+self.relation_runtime.contact.float().mean().detach()

    def get_door_experiment_contract(self):
        contract=super().get_door_experiment_contract()
        contract.update(door_hold=dict(self._hold_spec),auto_close=dict(self._door_cfg['autoClose']),
                        spawn_distance_range=list(self.cfg['env']['doorSpawnDistanceRange']),
                        door_hold_observation_size=38,num_people=1,door_asset=self._door_asset_file)
        return contract
