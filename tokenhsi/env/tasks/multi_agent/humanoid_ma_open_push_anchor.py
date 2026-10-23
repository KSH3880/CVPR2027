"""Scratch PUSH with owner-bound material anchors in the 30-D object view."""
import copy
import torch
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import PushOnlyRuntime,push_reward
from utils.push_point import validate_point,local_points,world_points,face_contact,face_phi,point_geometry,anchor_view


class PushPointRuntime(PushOnlyRuntime):
    def __init__(self,*args,point_spec,**kwargs):
        super().__init__(*args,**kwargs)
        self.point_spec=point_spec
        self.point_local=torch.zeros_like(self.start)

    def _push_phi(self,hands,boxes):
        return face_phi(hands,boxes,self.size,self.face_side,self.point_local)

    def step(self,roots,hands,handles,angles,boxes,goals):
        p=push_reward(roots,hands,boxes,goals,self.start,self.push_eligible,self.config,self.size,
            self.push_contact,self.push_credit,self.push_reset_distance,self.goal_contact,self.face_side)
        old_phi=p['push_phi']
        phi=self._push_phi(hands,boxes)
        saturated=self.push_contact|p['success']
        correction=.5*self.config['state_reward_weight']*torch.where(saturated,0.,phi-old_phi)
        state=p['state']+correction
        total=p['total']+correction
        self.phi[:,::2]=phi;self.phi[:,1::2]=p['phi']
        self.own_success[:,::2]=self.push_contact;self.own_success[:,1::2]=p['success']
        self.done.copy_(p['success']);self.achieved|=self.own_success;self.error.copy_(p['error'])
        return dict(state=state,progress=p['progress'],bonus=p['bonus'],
            reward=.9*total+.1*total.flip(-1),success=p['success'])


class HumanoidMAOpenPushAnchor(HumanoidMAOpenPush):
    _runtime_class=PushPointRuntime

    def __init__(self,cfg,*args,**kwargs):
        self._point_spec=validate_point(cfg['env'])
        # AMP keeps all source experts; RSI draws only physically screened poses.
        self._allow_partial_push_rsi=True
        super().__init__(cfg,*args,**kwargs)

    def get_clean_object_obs_size(self):return 30

    def get_open_push_contract(self):
        contract=super().get_open_push_contract()
        contract.update(policy_observation='push_contact_anchor_view_30d_v2',object_obs_size=30,
            push_objective='fixed_material_face_no_midpoint_v1',
            push_contact_point=copy.deepcopy(self._point_spec))
        return contract

    def _init_relation_runtime(self):
        super()._init_relation_runtime()
        self.relation_runtime=self._runtime_class(self.num_envs,self._relation_graph_spec,self._relation_cfg,
            self.dt,self.device,self.cfg['env']['pushBox']['size'],True,point_spec=self._runtime_point_spec())
        self._point_heights_physical=torch.zeros(self.num_envs,2,device=self.device)

    def _runtime_point_spec(self):
        return self._point_spec

    def _reset_boxes(self,env_ids):
        super()._reset_boxes(env_ids)
        lo,hi=self._point_spec['height_range_m']
        self._point_heights_physical[env_ids]=lo+(hi-lo)*torch.rand(len(env_ids),2,device=self.device)
        if getattr(self,'_point_height_override',None) is not None:
            self._point_heights_physical[env_ids]=self._point_height_override

    def _push_rsi_height(self,env,agent):
        return self._point_heights_physical[env,self._push_binding[env,agent]]

    def _reset_task(self,env_ids):
        super()._reset_task(env_ids)
        runtime=self.relation_runtime
        height=self._owner_push_values(self._point_heights_physical,env_ids)
        runtime.point_local[env_ids]=local_points(runtime.face_side[env_ids],self._owner_push_sizes(env_ids),height)

    def contact_target_points(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        return world_points(self._owner_push_boxes(env_ids),self.relation_runtime.point_local[ids])

    def _compute_clean_scene_obs(self,env_ids=None):
        obs=super()._compute_clean_scene_obs(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        return anchor_view(obs,self.num_agents,self.get_scene_entity_sizes(),self.relation_runtime.point_local[ids])

    def render(self,sync_frame_time=False):
        if self.viewer and hasattr(self,'relation_runtime') and self.relation_runtime.face_side is not None:
            self.gym.clear_lines(self.viewer)
            points=self.contact_target_points().detach().cpu().numpy()
            import numpy as np
            offsets=np.eye(3,dtype=np.float32)*.045
            for e in range(min(self.num_envs,16)):
                for owner in range(2):
                    vertices=np.stack((points[e,owner]-offsets,points[e,owner]+offsets),1).reshape(3,6)
                    color=np.tile(np.array([[1.,.2,.2] if owner==0 else [.2,.4,1.]],dtype=np.float32),(3,1))
                    self.gym.add_lines(self.viewer,self.envs[e],3,vertices,color)
        return super().render(sync_frame_time)

    def _physics_step(self):
        for _ in range(self.control_freq_inv):
            self.render();self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True);self._refresh_sim_tensors()
            hands=self._rigid_body_pos[:,:,self._key_body_ids[:2]]
            forces=self._contact_forces[:,:,self._key_body_ids[:2]]
            runtime=self.relation_runtime;boxes=self._owner_push_boxes()
            hit=self._push_contact_hit(hands,forces,boxes)
            self._push_hits|=hit
            runtime.accumulate_push_credit(boxes,self._tar_pos,hit)

    def _push_contact_hit(self,hands,forces,boxes):
        runtime=self.relation_runtime
        return face_contact(hands,forces,boxes,self._owner_push_values(self._box_forces),self._owner_push_sizes(),
            self._relation_cfg['push'],runtime.face_side,runtime.point_local)

    def _compute_reward(self,actions):
        super()._compute_reward(actions)
        boxes=self._owner_push_boxes()
        target,_,distance,_=point_geometry(self._rigid_body_pos[:,:,self._key_body_ids[:2]],
            boxes,self._owner_push_sizes(),self.relation_runtime.face_side,self.relation_runtime.point_local)
        # Replace legacy0.8m diagnostic reference with the sampled material point.
        old_z=boxes[...,2]+self._relation_cfg['push']['hand_target_height']-self._owner_push_sizes()[...,2]/2
        for hand in ('right','left'):
            self._diag['push/physics/'+hand+'_height_error_sum']+=(old_z-target[...,2]).sum().detach()
        for name,value in dict(face_distance_m=distance.amax(-1),
                               anchor_target_z_m=target[...,2]).items():
            key='push/physics/'+name+'_sum'
            self._diag[key]=self._diag.get(key,0)+value.sum().detach()
