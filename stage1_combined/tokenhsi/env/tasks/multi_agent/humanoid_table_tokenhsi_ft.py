"""Original Table reward/reset with our 78k scene observation architecture."""
import os
import torch
from env.tasks.multi_agent.humanoid_table_hold_at_ft import HumanoidTableHoldAtFT
from utils.hold_at_ft import object_catalog
from utils.tokenhsi_table_kernels import (compute_walk_reward,compute_handheld_reward,
    compute_carry_reward,compute_putdown_reward,compute_straight_upperbody_penalty,
    compute_humanoid_reset)


class HumanoidTableTokenHSIFT(HumanoidTableHoldAtFT):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.REWARD_TERM_NAMES=('walk','handheld','carry','putdown','upperbody','power','total')
        self._native_head=self.gym.find_actor_rigid_body_handle(self.envs[0],self.humanoid_handles[0],'head')

    def _load_box_asset(self):
        super()._load_box_asset()
        cfg=self.cfg['env']['holdAtFT']
        cat=object_catalog(os.environ.get('HOLD_AT_OBJECT_ROOT',cfg['object_root']),cfg['asset_split'],cfg['category'])
        self._native_up=torch.tensor([cat[i]['up_facing'] for i in self._object_model_ids.cpu().tolist()],device=self.device,dtype=torch.float)

    def _reset_boxes(self,ids):
        super()._reset_boxes(ids)
        if not hasattr(self,'_native_source'):
            self._native_source=torch.zeros(self.num_envs,3,device=self.device)
        self._native_source[ids]=self._box_states[ids,0,:3]

    def _compute_reward(self,actions):
        # Relation runtime remains available solely for graph input and diagnostics.
        old_sums=self._reward_term_sums.clone();old_count=self._reward_term_count
        super()._compute_reward(actions)
        root=self._humanoid_root_states[:,0];box=self._box_states[:,0]
        body=self._rigid_body_pos[:,0];target=self._tar_pos[:,0];size=self._box_size[:,0]
        walk=compute_walk_reward(root[:,:3],self._prev_root_pos[:,0],box[:,:3],self.dt,1.5,True)
        held=compute_handheld_reward(body,box[:,:3],self._key_body_ids[:2],False)
        carry=compute_carry_reward(box[:,:3],self._prev_box_pos[:,0],target,self.dt,1.5,size,
            True,True,1.,2.,box[:,3:7],self._native_up,.8,True,self._native_source,
            root[:,:3],self._prev_root_pos[:,0])
        down=compute_putdown_reward(box[:,:3],target,box[:,3:7],size[:,2],self._box_bps[:,0],body)
        upper=compute_straight_upperbody_penalty(box[:,:3],size,self._native_source,target,
            root[:,:3],root[:,3:7],body[:,self._native_head])
        zero=torch.zeros_like(walk)
        parts=torch.stack((.2*walk,.2*held,carry,.2*down,upper,zero),-1)
        total=parts.sum(-1);terms=torch.cat((parts,total[:,None]),-1)
        self.rew_buf.copy_(total);self.extras['reward_terms']=terms
        self._reward_term_sums.copy_(old_sums+terms.sum(0));self._reward_term_count=old_count+self.num_envs

    def _compute_reset(self):
        self.reset_buf[:],self._terminate_buf[:]=compute_humanoid_reset(self.reset_buf,self.progress_buf,
            self._contact_forces[:,0],self._contact_body_ids,self._rigid_body_pos[:,0],
            float(self.max_episode_length),self._enable_early_termination,self._termination_heights,
            self._box_states[:,0],True,self._box_size[:,0,2],self._native_source,
            self._tar_pos[:,0],self._humanoid_root_states[:,0,7:10])
