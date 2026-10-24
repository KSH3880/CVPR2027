"""Partial physical scene with RawCuboid baseline contact and common-goal reward."""
import torch
from isaacgym import gymtorch
from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi
from env.tasks.multi_agent.push_reward import contact_geometry
from utils.motion_lib import MotionLib
from .scene_base import PartitionSceneBase
from .reward import components,rotate


class PartialPushScene(PartitionSceneBase):
    _amp_num_families=4
    _solo_cache_class=LocoRsi
    TERMS=('contact_state','at_state','contact_progress','at_progress','contact_success','at_success','power','collision','total')

    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        self.REWARD_TERM_NAMES=self.TERMS
        self._reward_term_sums=torch.zeros(len(self.TERMS),device=self.device)
        self._reward_term_count=0
        self._push_hits=torch.zeros(self.num_envs,4,dtype=torch.bool,device=self.device)
        forces=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        start=self.num_agents*self.num_bodies
        self._physical_box_forces=forces.view(self.num_envs,-1,3)[:,start:start+24]
        self._partial_diagnostics={};self._partial_diag_count=0

    def _load_motion(self,motion_file):
        args=dict(motion_file=motion_file,dof_body_ids=self._dof_body_ids,dof_offsets=self._dof_offsets,key_body_ids=self._key_body_ids.cpu().numpy(),device=self.device)
        loco=MotionLib(skill='push_loco',**args)
        self._motion_lib={'loco':loco,'push_loco':loco,'push':MotionLib(skill='push',**args)}
        self._skill_categories=['loco','push_loco','push']

    def _fetch_conditioned_amp_demo(self,count):
        from utils.unified_training import append_family
        result=torch.empty(count,self.get_num_amp_obs(),device=self.device)
        push=torch.rand(count,device=self.device)>=.33
        for selected,skill in ((~push,'push_loco'),(push,'push')):
            ids=selected.nonzero().flatten()
            if not len(ids):continue
            lib=self._motion_lib[skill];clips=lib.sample_motions(len(ids));history=self.dt*9
            times=lib.sample_time(clips,truncate_time=history)+history
            obs=self.build_amp_obs_demo(clips,times,lib).reshape(len(ids),10,-1)
            result[ids]=append_family(obs,torch.full((len(ids),10),3,device=self.device,dtype=torch.long),4).flatten(1)
        return result

    def _select_regions(self,ids,live=False,reassign=False):
        super()._select_regions(ids,live=live,reassign=reassign)
        size=self._box_size[ids[:,None],self._agent_box_assignment[ids]]
        self._local_hands[ids,:,:,2]=(self.cfg['env']['partialReward']['contact_height']-size[...,2]/2)[...,None]

    def _contact_hit(self):
        rows=torch.arange(self.num_envs,device=self.device)[:,None];a=self._agent_box_assignment
        box=self._box_states[rows,a].clone();size=self._box_size[rows,a].clone()
        offset=torch.zeros_like(box[...,:3]);offset[...,0]=self._local_hands[...,0].mean(-1)
        box[...,:3]+=rotate(box[...,3:7],offset);size[...,0]/=self._group_counts
        return contact_geometry(self._rigid_body_pos[:,:,self._key_body_ids[:2]],box,size,self._contact_forces[:,:,self._key_body_ids[:2]],self._physical_box_forces[rows,a],midpoint_only=True,contact_height=self.cfg['env']['partialReward']['contact_height'])[1]

    def pre_physics_step(self,actions):
        super().pre_physics_step(actions)
        self._push_hits.zero_()

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            if hasattr(self,'_push_hits'):self._push_hits |= self._contact_hit()

    def _compute_reward(self,actions):
        self._select_regions(torch.arange(self.num_envs,device=self.device),live=True)
        r=components(self._humanoid_root_states,self._rigid_body_pos[:,:,self._key_body_ids[:2]],self._contact_forces[:,:,self._key_body_ids[:2]],self._box_states,self._box_size,self._physical_box_forces,self._agent_box_assignment,self._local_hands,self._tar_pos,self.cfg['env']['partialReward'],contact_geometry,self._push_hits)
        self.relation_runtime.phi.copy_(r['phi'].flatten(1));self.relation_runtime.own_success.copy_((r['success']>0).flatten(1));self.relation_runtime.achieved.copy_(self.relation_runtime.own_success);self.relation_runtime.done.copy_(r['done'])
        power=-self._power_coefficient*(self.dof_force_tensor*self._dof_vel).abs().sum(-1) if self._power_reward else torch.zeros_like(r['reward'])
        roots=self._humanoid_root_states[...,:2]
        distances=torch.cdist(roots,roots);distances.masked_fill_(torch.eye(4,device=self.device,dtype=torch.bool),torch.inf)
        collision=-self._agent_collision_coeff*((self._agent_collision_dist-distances.amin(-1))/self._agent_collision_dist).clamp_min(0) if self._agent_collision_penalty else torch.zeros_like(power)
        total=r['reward']+power+collision;self.rew_buf[:]=total.flatten()
        terms=torch.cat((r['state'],r['progress'],r['success'],power[...,None],collision[...,None],total[...,None]),-1).reshape(-1,len(self.TERMS))
        self.extras['reward_terms']=terms;self._reward_term_sums+=terms.sum(0);self._reward_term_count+=len(terms)
        self.extras['partial_group_gate']=r['group_gate'];self.extras['partial_goal_distance']=r['goal_distance']
        for key,value in {'group_gate':r['group_gate'],'hand_face':r['hand_face'],'contact':r['contact'],'at_success':r['done'],'goal_distance':r['goal_distance'],'raw_at_progress':r['raw_at_progress']}.items():
            self._partial_diagnostics[key]=self._partial_diagnostics.get(key,0)+value.float().mean().detach()
        self._partial_diag_count+=1

    def consume_relation_diagnostics(self):
        count=max(self._partial_diag_count,1)
        result={'partial_push/'+k:v/count for k,v in self._partial_diagnostics.items()}
        self._partial_diagnostics={};self._partial_diag_count=0
        return result
