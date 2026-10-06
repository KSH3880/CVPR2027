"""Four physical boxes, two frozen-ms18 executors, completion-time scheduling.

The parent's native two-box tensor remains a simulator alias for reset. Only
read-only observation/reward/planner contexts gather the two selected objects.
No actor is teleported when a job is assigned.
"""
import os
from contextlib import contextmanager
from isaacgym import gymapi
import numpy as np
import torch
from carry_box_swap.ms18_env import HumanoidMACarryBoxSwapMS18
from carry_planner.env_adapter import HumanoidMACarryPlannerTrain
from carry_planner.held_state import observed_box_held
from task_allocation.core import sample_layout


class HumanoidFourBoxAllocationMS18(HumanoidMACarryBoxSwapMS18):
    def _build_box(self, env_id, env_ptr):
        super()._build_box(env_id, env_ptr)
        if not hasattr(self, '_extra_box_handles'):
            self._extra_box_handles=[]
        handles=[]
        for extra in range(2):
            row=env_id*2+extra
            pose=gymapi.Transform()
            pose.p.x=5.; pose.p.y=4.+2.*extra
            pose.p.z=float(self._box_lib._box_size[row, 2])*.5
            handles.append(self.gym.create_actor(env_ptr, self._box_assets[row], pose,
                                                f'box{extra+2}', env_id, 0, 0))
        # Parent native boxes stay at the first two slots. Extras must be adjacent.
        assert handles == [self._box_handles[-1][0]+2, self._box_handles[-1][0]+3]
        self._extra_box_handles.append(handles)

    def _ensure_jobs(self):
        if not hasattr(self, 'job_assignment'):
            self.job_assignment=torch.full((self.num_envs, 2), -1, dtype=torch.long, device=self.device)
            self.allocation_delivered=torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
            self.allocation_locked=torch.zeros_like(self.allocation_delivered)
            self.allocation_stable=torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
            self.allocation_initialized=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _ensure_pool(self):
        if hasattr(self, '_allocation_boxes'):
            return True
        if not hasattr(self, '_box_states'):
            return False
        actors=self.get_num_actors_per_env()
        first=self._box_handles[0][0]
        assert first+4 <= actors
        self._allocation_boxes=self._root_states.view(self.num_envs, actors, 13)[:, first:first+4]
        size=self._box_lib._box_size.reshape(self.num_envs, 2, 3)
        bps=self._box_lib._box_bps.reshape(self.num_envs, 2, 8, 3)
        self._allocation_sizes=size.repeat(1, 2, 1).clone()
        self._allocation_bps=bps.repeat(1, 2, 1, 1).clone()
        self._allocation_goals=torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._allocation_previous=self._allocation_boxes[..., :3].clone()
        return True

    @contextmanager
    def _assigned(self):
        self._ensure_assignment()
        if not self._ensure_pool():
            with super()._assigned():
                yield
            return
        if getattr(self, '_swap_inputs_active', False):
            yield
            return
        e=torch.arange(self.num_envs, device=self.device)[:, None]
        index=self.box_assignment
        saved=(self._box_states,self._box_tar_pos,self._prev_box_pos,
               self._box_lib._box_size,self._box_lib._box_bps)
        self._swap_inputs_active=True
        try:
            self._box_states=self._allocation_boxes[e,index]
            self._box_tar_pos=self._allocation_goals[e,index].reshape(-1,3)
            self._prev_box_pos=self._allocation_previous[e,index].reshape(-1,3)
            self._box_lib._box_size=self._allocation_sizes[e,index].reshape(-1,3)
            self._box_lib._box_bps=self._allocation_bps[e,index].reshape(-1,8,3)
            yield
        finally:
            (self._box_states,self._box_tar_pos,self._prev_box_pos,
             self._box_lib._box_size,self._box_lib._box_bps)=saved
            self._swap_inputs_active=False

    def _reset_envs(self, env_ids):
        self._ensure_jobs()
        self.job_assignment[env_ids]=-1
        self.allocation_delivered[env_ids]=False
        self.allocation_locked[env_ids]=False
        self.allocation_stable[env_ids]=0
        self.allocation_initialized[env_ids]=False
        super()._reset_envs(env_ids)
        if hasattr(self, '_carry_timeout_deadline'):
            self._carry_timeout_deadline[env_ids]=self.max_episode_length

    def apply_layout(self, env_ids):
        if not len(env_ids):
            return
        self._ensure_jobs()
        assert self._ensure_pool()
        center=self.agent_axis(self._initial_humanoid_root_states)[env_ids,:, :2].mean(1)
        size=self._allocation_sizes[env_ids]
        clearance=max(float(os.environ.get('ALLOC_CLEARANCE','1.2')),
                      float(size[..., :2].norm(dim=-1).max())+.3)
        points=sample_layout(len(env_ids),self.device,float(os.environ.get('ALLOC_EXTENT','3')),
                             clearance,points=10)+center[:,None]
        self.agent_axis(self._humanoid_root_states)[env_ids,:, :2]=points[:, :2]
        self._allocation_boxes[env_ids,:,:2]=points[:,2:6]
        self._allocation_boxes[env_ids,:,2]=size[...,2]*.5+.05
        self._allocation_boxes[env_ids,:,3:7]=0
        self._allocation_boxes[env_ids,:,6]=1
        self._allocation_boxes[env_ids,:,7:13]=0
        self._allocation_goals[env_ids,:,:2]=points[:,6:10]
        self._allocation_goals[env_ids,:,2]=size[...,2]*.5
        self._allocation_previous[env_ids]=self._allocation_boxes[env_ids,:,:3]
        # Original reset/progress buffers still address the physical first two boxes.
        self._box_tar_pos[self.agent_rows(env_ids)]=self._allocation_goals[env_ids,:2].reshape(-1,3)
        if self._carry_reset_random_height:
            raise ValueError('four-box scheduling requires floor boxes, randomHeight=False')

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        if self._ensure_pool():
            self._allocation_previous.copy_(self._allocation_boxes[..., :3])

    def set_box_assignment(self, env_ids, assignment):
        self._ensure_jobs()
        if assignment.shape!=(len(env_ids),2) or assignment.dtype!=torch.long:
            raise ValueError('assignment must be long [envs,2]')
        if ((assignment < -1) | (assignment > 3)).any():
            raise ValueError('invalid box id')
        if ((assignment[:,0]>=0) & (assignment[:,0]==assignment[:,1])).any():
            raise ValueError('duplicate box owner')
        previous=self.job_assignment[env_ids]
        if ((previous>=0) & (assignment!=previous)).any():
            raise ValueError('in-progress jobs cannot be reassigned')
        selected_done=self.allocation_delivered[env_ids].gather(1,assignment.clamp(min=0))
        if (selected_done & (assignment>=0)).any():
            raise ValueError('delivered jobs cannot be selected')
        starts=(previous<0) & (assignment>=0)
        self.job_assignment[env_ids]=assignment
        # Idle executor keeps its last completed box/goal and balances there.
        self.box_assignment[env_ids]=torch.where(assignment>=0,assignment,self.box_assignment[env_ids])
        rows=self.agent_rows(env_ids).reshape(-1,2)[starts]
        for name in ('_IET_step_buf','_IET_triggered_buf','_success_buf'):
            if hasattr(self,name):getattr(self,name)[rows]=0
        if hasattr(self,'_ep_finish'): self._ep_finish[rows]=-1
        self._coord_has_valid[env_ids]=False
        self._plan_envs(env_ids)

    def _physical_state(self):
        self._ensure_jobs(); assert self._ensure_pool()
        roots=self.humanoid_rows(self._humanoid_root_states).reshape(self.num_envs,2,13)
        body=self.humanoid_rows(self._rigid_body_pos).reshape(self.num_envs,2,-1,3)
        hands=body[:,:,self._key_body_ids[[0,1]]].mean(2)
        held_pairs=observed_box_held(
            roots[:,:,None,:2].expand(-1,-1,4,-1),
            self._allocation_boxes[:,None,:,:3].expand(-1,2,-1,-1),
            self._allocation_sizes[:,None,:,2].expand(-1,2,-1),
            hands[:,:,None,:].expand(-1,-1,4,-1))
        return roots,held_pairs.any(1)

    def allocation_observation(self):
        roots,held=self._physical_state()
        root,box,goal=roots.clone(),self._allocation_boxes.clone(),self._allocation_goals.clone()
        center=self.agent_axis(self._initial_humanoid_root_states)[...,:2].mean(1)
        root[...,:2]-=center[:,None];box[...,:2]-=center[:,None];goal[...,:2]-=center[:,None]
        phase=self.planner_state().phase/3
        chosen_held=held.gather(1,self.box_assignment)
        remaining=(1-self.progress_buf.float()/self.max_episode_length)[:,None,None].expand(-1,2,1)
        agent=torch.cat((root,chosen_held[...,None].float(),phase[...,None],remaining),-1)
        owners=torch.nn.functional.one_hot(self.job_assignment.clamp(min=0),4).float()
        owners*= (self.job_assignment>=0)[...,None]
        owners=owners.transpose(1,2)
        features=torch.cat((box,self._allocation_sizes,owners,held[...,None].float(),
                            self.allocation_delivered[...,None].float()),-1)
        return dict(agent=agent,box=features,goal=goal,assignment=self.job_assignment.clone(),
                    delivered=self.allocation_delivered.clone(),locked=self.allocation_locked.clone())

    def update_delivery(self):
        _,held=self._physical_state()
        self.allocation_locked |= held
        placed=((self._allocation_boxes[...,:3]-self._allocation_goals).norm(dim=-1)<.3)
        placed &= (self._allocation_boxes[...,7:10].norm(dim=-1)<.2) & ~held & self.allocation_locked
        self.allocation_stable=torch.where(placed,self.allocation_stable+1,0)
        new=(self.allocation_stable>=10) & ~self.allocation_delivered
        self.allocation_delivered |= new
        done=self.allocation_delivered.gather(1,self.job_assignment.clamp(min=0))
        self.job_assignment=torch.where(done & (self.job_assignment>=0),-1,self.job_assignment)
        return new.sum(-1),self.allocation_delivered.all(-1)

    def _update_marker(self):
        # Do not re-submit physical box actors from render between simulations.
        return

    def _draw_task(self):
        with self._assigned():
            HumanoidMACarryPlannerTrain._draw_task(self)
        root=self.humanoid_rows(self._humanoid_root_states).reshape(self.num_envs,2,13)
        e=torch.arange(self.num_envs,device=self.device)[:,None]
        boxes=self._allocation_boxes[e,self.box_assignment]
        lines=torch.cat((root[...,:3],boxes[...,:3]),-1).cpu().numpy()
        colors=np.asarray([[0.,1.,1.],[1.,.5,0.]],dtype=np.float32)
        for i,env in enumerate(self.envs):
            self.gym.add_lines(self.viewer,env,2,lines[i],colors)
            for j in range(4):
                start=self._allocation_boxes[i,j,:3].cpu().numpy()
                end=self._allocation_goals[i,j].cpu().numpy()
                color=np.asarray([[.2,.8,.2] if not self.allocation_delivered[i,j] else [.5,.5,.5]],dtype=np.float32)
                self.gym.add_lines(self.viewer,env,1,np.concatenate((start,end))[None],color)
