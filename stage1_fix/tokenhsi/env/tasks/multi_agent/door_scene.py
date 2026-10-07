"""Reusable articulated door fixture; independent of policy/reward definitions."""
from pathlib import Path
from isaacgym import gymapi, gymtorch
import torch
from tokenhsi.utils.door_asset import DoorSpec

ROOT=Path(__file__).resolve().parents[4]
ASSET=ROOT/'tokenhsi/data/assets/door/left_hinge_right_handle.urdf'


class DoorFixture:
    def __init__(self,gym,sim,spec=DoorSpec(),asset_path=ASSET):
        self.gym,self.sim,self.spec=gym,sim,spec
        options=gymapi.AssetOptions()
        options.fix_base_link=True
        options.collapse_fixed_joints=False
        options.disable_gravity=False
        options.default_dof_drive_mode=gymapi.DOF_MODE_POS
        options.angular_damping=0.
        self.asset=gym.load_asset(sim,str(asset_path.parent),asset_path.name,options)
        if self.asset is None or gym.get_asset_dof_count(self.asset)!=1:
            raise ValueError('Door asset must have exactly one hinge DOF')
        self.body_names=['frame','panel','handle','handle_back']
        self.body_ids={n:gym.find_asset_rigid_body_index(self.asset,n) for n in self.body_names}
        if min(self.body_ids.values())<0:raise ValueError('Missing door body')
        properties=gym.get_asset_dof_properties(self.asset)
        if abs(float(properties['upper'][0])-spec.max_angle)>1e-5:
            raise ValueError('Door asset and DoorSpec opening limit disagree')
        self.properties=properties
        self.records=[]

    def add(self,env,collision_group,pose=None,name='door'):
        pose=pose or gymapi.Transform()
        # Disable self collisions within this articulation; other actors collide.
        actor=self.gym.create_actor(env,self.asset,pose,name,collision_group,1)
        properties=self.properties.copy()
        properties['driveMode']=gymapi.DOF_MODE_POS
        properties['stiffness']=self.spec.spring_stiffness
        properties['damping']=self.spec.spring_damping
        properties['friction']=0.
        self.gym.set_actor_dof_properties(env,actor,properties)
        self.gym.set_actor_dof_position_targets(env,actor,torch.zeros(1).numpy())
        shape_properties=self.gym.get_actor_rigid_shape_properties(env,actor)
        for prop in shape_properties:
            prop.friction=.8
            prop.restitution=0.
        self.gym.set_actor_rigid_shape_properties(env,actor,shape_properties)
        record=dict(env=env,actor=actor,
            actor_index=self.gym.get_actor_index(env,actor,gymapi.DOMAIN_SIM),
            dof_index=self.gym.get_actor_dof_index(env,actor,0,gymapi.DOMAIN_SIM),
            bodies={n:self.gym.get_actor_rigid_body_index(env,actor,b,gymapi.DOMAIN_SIM)
                    for n,b in self.body_ids.items()})
        self.records.append(record)
        return actor

    def bind_tensors(self):
        self.dof_states=gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.body_states=gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        device=self.dof_states.device
        self.dof_indices=torch.tensor([r['dof_index'] for r in self.records],device=device,dtype=torch.long)
        self.actor_indices=torch.tensor([r['actor_index'] for r in self.records],device=device,dtype=torch.int32)
        self.body_indices={n:torch.tensor([r['bodies'][n] for r in self.records],device=device,dtype=torch.long)
                           for n in self.body_names}
        self.refresh()

    def refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def observe(self):
        states=self.dof_states[self.dof_indices]
        return dict(angle=states[:,0].clone(),angular_velocity=states[:,1].clone(),
                    panel_pose=self.body_states[self.body_indices['panel'],:7].clone(),
                    handle_position=self.body_states[self.body_indices['handle'],:3].clone(),
                    back_handle_position=self.body_states[self.body_indices['handle_back'],:3].clone(),
                    base_pose=self.body_states[self.body_indices['frame'],:7].clone())

    def reset(self,door_ids=None,angle=0.):
        """Reset selected hinges; simulate/refresh before reading new body poses."""
        ids=torch.arange(len(self.records),device=self.dof_states.device) if door_ids is None else torch.as_tensor(door_ids,device=self.dof_states.device,dtype=torch.long)
        if not 0 <= angle <= self.spec.max_angle:raise ValueError('Reset angle outside hinge limits')
        indices=self.dof_indices[ids]
        self.dof_states[indices,0]=angle
        self.dof_states[indices,1]=0.
        actors=self.actor_indices[ids].contiguous()
        if not self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_states),
                                                    gymtorch.unwrap_tensor(actors),len(actors)):
            raise RuntimeError('Door DOF reset failed')
        # The zero-position drive remains a compliant self-closing spring.
