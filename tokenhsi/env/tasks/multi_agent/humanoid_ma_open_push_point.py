"""Original center observations with a fixed material-point PUSH objective."""
import copy
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.humanoid_ma_open_push_anchor import HumanoidMAOpenPushAnchor,PushPointRuntime
from utils.push_point import validate_point_reward,point_reward_phi,point_reward_contact,point_geometry,point_height_excess,point_height_valid


class PushPointRewardRuntime(PushPointRuntime):
    def _push_phi(self,hands,boxes):
        return point_reward_phi(hands,boxes,self.size,self.face_side,self.point_local,self.point_spec)


class HumanoidMAOpenPushPoint(HumanoidMAOpenPushAnchor):
    _runtime_class=PushPointRewardRuntime

    def __init__(self,cfg,*args,**kwargs):
        self._point_reward_spec=validate_point_reward(cfg['env'])
        super().__init__(cfg,*args,**kwargs)

    def _runtime_point_spec(self):
        return self._point_reward_spec

    def get_open_push_contract(self):
        contract=HumanoidMAOpenPush.get_open_push_contract(self)
        contract.update(policy_observation='push_no_doors_30d_v2',object_obs_size=30,
            push_objective=('fixed_material_point_height_band_v2' if 'height_band_m' in self._point_reward_spec
                            else 'fixed_material_point_midpoint_v1'),
            push_contact_point=copy.deepcopy(self._point_spec),
            push_point_reward=copy.deepcopy(self._point_reward_spec))
        return contract

    def _compute_clean_scene_obs(self,env_ids=None):
        return HumanoidMAOpenPush._compute_clean_scene_obs(self,env_ids)

    def _push_contact_hit(self,hands,forces,boxes):
        runtime=self.relation_runtime
        return point_reward_contact(hands,forces,boxes,self._owner_push_values(self._box_forces),
            self._owner_push_sizes(),self._relation_cfg['push'],runtime.face_side,runtime.point_local,
            self._point_reward_spec)

    def _compute_reward(self,actions):
        super()._compute_reward(actions)
        runtime=self.relation_runtime
        _,delta,_,_=point_geometry(self._rigid_body_pos[:,:,self._key_body_ids[:2]],self._owner_push_boxes(),
            self._owner_push_sizes(),runtime.face_side,runtime.point_local)
        for name,value in dict(point_distance_m=delta.norm(dim=-1),point_z_error_m=point_height_excess(delta,self._point_reward_spec),
                point_z_in_tolerance=point_height_valid(delta,self._point_reward_spec).float()).items():
            key='push/physics/'+name+'_sum'
            self._diag[key]=self._diag.get(key,0)+value.sum().detach()
