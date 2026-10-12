"""Shared two-person PUSH+AT, loco-only RSI, original conditioned Push AMP."""
import hashlib
from pathlib import Path
from isaacgym import gymapi
import torch
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.joint_push_reward import joint_push_reward
from utils.door_reference import rotate, multiply, yaw_from_quat
from utils.door_hold import collect_foot_support


def cpa_penalty(pos, vel, distance, discount, dt):
    p=pos[:,:,None,:2]-pos[:,None,:,:2];v=vel[:,:,None,:2]-vel[:,None,:,:2]
    dot=(p*v).sum(-1)
    closing=(-dot/(p.norm(dim=-1)*v.norm(dim=-1)).clamp_min(1e-6)).clamp(0,1)
    time=(-dot/v.square().sum(-1).clamp_min(1e-6)).clamp_min(0)
    risk=((distance-(p+time[...,None]*v).norm(dim=-1))/distance).clamp(0,1)
    penalty=closing*risk*discount**(time/dt)
    return penalty.masked_fill(torch.eye(2,device=p.device,dtype=torch.bool),0).amax(-1)


class HumanoidMAJointPush(HumanoidMAOpenPush):
    def __init__(self,cfg,*args,**kwargs):
        self._joint_push=True
        spec=cfg['env']['jointPush']
        if cfg['env']['numAgents']!=2 or cfg['env']['numObjects']!=4:
            raise ValueError('Joint Push requires two agents and four observation object slots')
        if cfg['env'].get('openPushRSI') or cfg['env']['skillInitProb']!=[1.,0.,0.]:
            raise ValueError('Joint Push requires loco-only RSI')
        if spec['rsi']!='loco_only' or spec['amp_history']!='repeat_reset_state':
            raise ValueError('Unsupported joint Push initialization')
        self._joint_push_spec=spec
        from env.tasks.multi_agent.joint_push_reward import validate_config
        validate_config(spec['reward'])
        if cfg['env']['relationReward'].get('joint_push_reward')!=spec['reward']:
            raise ValueError('Effective reward metadata differs')
        if spec['collision'] != {'mode':'cpa','ttc_discount':.99}:
            raise ValueError('Joint Push requires Carry CPA collision contract')
        super().__init__(cfg,*args,**kwargs)
        import os
        if os.environ.get('JOINT_PUSH_AUDIT_PATH'):
            from utils.joint_push_audit import audit_runtime
            audit_runtime(self,os.environ['JOINT_PUSH_AUDIT_PATH'])

    def get_open_push_contract(self):
        contract=super().get_open_push_contract()
        contract.update(joint_push=self._joint_push_spec,
                        push_reward_contract='joint_push_symmetric_midpoints_v1',
                        policy='push_at_ca4',shared_payload=2,shared_goal=0)
        return contract

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self._push_binding.zero_()

    def _sample_tasks(self,env_ids):
        return torch.ones(len(env_ids),2,device=self.device,dtype=torch.long)

    def _sample_box_assignments(self,env_ids):
        self._logical_box_order[env_ids]=torch.arange(4,device=self.device)
        self._agent_box_assignment[env_ids]=2
        graph=self.relation_runtime.graph
        graph.edge_src[env_ids]=torch.tensor([0,4,1,4],device=self.device)
        graph.edge_dst[env_ids]=torch.tensor([4,6,4,6],device=self.device)
        graph.edge_relation[env_ids]=torch.tensor([12,7,12,7],device=self.device)
        graph.edge_valid[env_ids]=True
        graph.required_goal[env_ids]=torch.tensor([False,True,False,True],device=self.device)

    def _reset_boxes(self,env_ids):
        super()._reset_boxes(env_ids)
        yaw=torch.rand(len(env_ids),device=self.device)*2*torch.pi
        box=self._box_states[env_ids,2].clone();box[:,:2]=0;box[:,2]=self.cfg['env']['pushBox']['size'][2]/2
        box[:,3:7]=0;box[:,5]=torch.sin(yaw/2);box[:,6]=torch.cos(yaw/2);box[:,7:]=0
        self._box_states[env_ids,2]=box
        self._box_states[env_ids,3,:2]=box.new_tensor([5.,5.])
        self._box_states[env_ids,:2,:2]=box.new_tensor([[5.,-5.],[-5.,5.]])
        self._push_reset_states[env_ids]=self._box_states[env_ids,2:]
        # All directions, matching Goal360; fixed goal survives contact changes.
        angle=torch.rand(len(env_ids),device=self.device)*2*torch.pi
        lo,hi=self._joint_push_spec['goal_distance']
        distance=lo+(hi-lo)*torch.rand(len(env_ids),device=self.device)
        offset=torch.stack((distance*angle.cos(),distance*angle.sin(),torch.zeros_like(distance)),-1)
        self._push_goal_offsets[env_ids]=offset[:,None]

    def _reset_task(self,env_ids):
        target=self._box_states[env_ids,2,:3]+self._push_goal_offsets[env_ids,0]
        self._tar_pos[env_ids]=target[:,None]

    def _reset_ref_state_init(self,slot_env,slot_agent):
        HumanoidMACarry._reset_ref_state_init(self,slot_env,slot_agent)
        root=self._humanoid_root_states[slot_env,slot_agent].clone()
        body=self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent].clone()
        from env.tasks.multi_agent.joint_push_reward import push_face_geometry
        boxes=self._box_states[slot_env,2]
        goals=boxes[:,:3]+self._push_goal_offsets[slot_env,0]
        center,normal,tangent,half_width,_=push_face_geometry(boxes,goals,self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])
        facing=torch.atan2(normal[:,1],normal[:,0]);turn=facing-yaw_from_quat(root[:,3:7])
        q=torch.zeros_like(root[:,3:7]);q[:,2]=torch.sin(turn/2);q[:,3]=torch.cos(turn/2)
        lo,hi=self._joint_push_spec['spawn_distance']
        distance=lo+(hi-lo)*torch.rand(len(slot_env),device=self.device)
        side=(slot_agent.to(root.dtype)*2-1)*half_width*.5
        target=root[:,:3].clone();target[:,:2]=center[:,:2]-normal*distance[:,None]+tangent*side[:,None]
        expanded=q[:,None].expand_as(body[...,3:7])
        body[...,:3]=rotate(expanded,body[...,:3]-root[:,None,:3])+target[:,None]
        body[...,3:7]=multiply(expanded,body[...,3:7]);body[...,7:10]=rotate(expanded,body[...,7:10]);body[...,10:13]=rotate(expanded,body[...,10:13])
        self._set_env_state(slot_env,slot_agent,target,multiply(q,root[:,3:7]),self._dof_pos[slot_env,slot_agent].clone(),
            rotate(q,root[:,7:10]),rotate(q,root[:,10:13]),self._dof_vel[slot_env,slot_agent].clone())
        self._kinematic_humanoid_rigid_body_states[slot_env,slot_agent]=body

    def _reset_envs(self,env_ids):
        if not len(env_ids):return
        self._reset_default_slots=None;self._reset_ref_slots={};self._reset_ref_motion_ids={};self._reset_ref_motion_times={}
        self._reset_boxes(env_ids);self._agent_lanes[env_ids]=torch.arange(2,device=self.device);self._push_binding[env_ids]=0
        self.relation_runtime.reset(env_ids,self._sample_tasks(env_ids),self._owner_push_boxes(env_ids))
        self._sample_box_assignments(env_ids);self._reset_actors(env_ids);self._reset_task(env_ids)
        self._reset_env_tensors(env_ids);self._sync_door_reset_states(env_ids)
        self._compute_observations(env_ids);self._init_amp_obs(env_ids)
        self._compute_amp_observations(env_ids)
        self._hist_amp_obs_buf[env_ids]=self._curr_amp_obs_buf[env_ids,:,None]
        self._update_joint_colors(env_ids)

    def _update_joint_colors(self,ids):
        if self.viewer is None:return
        for eid in ids.cpu().tolist():
            for o,handle in enumerate(self._box_handles[eid*4:eid*4+4]):
                color=gymapi.Vec3(1.,.8,.1) if o==2 else gymapi.Vec3(.45,.45,.45)
                for b in range(self._door_body_count if o<2 else 1):
                    self.gym.set_rigid_body_color(self.envs[eid],handle,b,gymapi.MESH_VISUAL,color)

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            collect_foot_support(self)

    def _compute_reward(self,actions):
        r=joint_push_reward(self._humanoid_root_states,self._rigid_body_pos[:,:,self._key_body_ids[:2]],
            self._box_states[:,2],self._tar_pos[:,0],self.cfg['env']['pushBox']['size'],self._joint_push_spec['reward'])
        runtime=self.relation_runtime
        runtime.phi.copy_(r['phi_raw'].reshape(self.num_envs,4))
        runtime.own_success.copy_(r['own_success'].reshape(self.num_envs,4));runtime.achieved.copy_(runtime.own_success)
        runtime.done.copy_(r['done']);runtime.error.copy_((self._box_states[:,2,:2]-self._tar_pos[:,0,:2]).norm(dim=-1)[:,None])
        power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(r['agent_task_reward'])
        collision=-self._agent_collision_coeff*cpa_penalty(self._humanoid_root_states[...,:3],self._humanoid_root_states[...,7:10],self._agent_collision_dist,.99,self.dt) if self._agent_collision_penalty else torch.zeros_like(power)
        total=r['agent_task_reward']+power+collision;self.rew_buf[:]=total.flatten()
        terms=torch.stack((r['state_component'].sum(-1),r['progress_component'].sum(-1),r['success_component'].sum(-1),power,collision,total),-1).reshape(-1,6)
        self.extras['reward_terms']=terms;self._reward_term_sums+=terms.sum(0);self._reward_term_count+=len(terms)
        self.extras['joint_push_gate']=r['coupled_contact'].clone()
        self.extras['joint_push_symmetry']=r['symmetry_phi'].clone()
        self.extras['joint_push_separation']=r['separation'].clone()
        self.extras['joint_push_box_pos']=self._box_states[:,2,:3].clone()

    def consume_relation_diagnostics(self):
        return {}
