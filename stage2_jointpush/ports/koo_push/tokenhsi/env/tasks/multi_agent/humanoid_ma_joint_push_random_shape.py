"""JointPush with true cuboid assets of different sizes, independent random mass per payload."""
import os
import math
from isaacgym import gymapi
import torch
from env.tasks.multi_agent.humanoid_ma_joint_push import HumanoidMAJointPush
from env.tasks.multi_agent.joint_push_shape import shape_mass_grid


class HumanoidMAJointPushRandomShape(HumanoidMAJointPush):
    def __init__(self,cfg,*args,**kwargs):
        self._shape_spec=cfg['env']['jointPush']['shape_randomization']
        self._shape_grid=shape_mass_grid(self._shape_spec)
        # Viewer-only selection from the checkpoint's actual training grid.
        view_index=os.environ.get('JOINT_PUSH_VIEW_GRID_INDEX')
        if view_index is not None:
            if not cfg['args'].test:
                raise ValueError('View grid selection is test-only')
            index=int(view_index)
            if not 0<=index<len(self._shape_grid):
                raise ValueError('View grid index outside checkpoint training grid')
            self._shape_grid=[self._shape_grid[index]]
            print('[JointPush view training-grid selection]',self._shape_grid[0],flush=True)
        preview_size=os.environ.get('JOINT_PUSH_PREVIEW_BOX_SIZE')
        preview_mass=os.environ.get('JOINT_PUSH_PREVIEW_BOX_MASS')
        if preview_size is not None or preview_mass is not None:
            two_preview=os.environ.get('JOINT_PUSH_TWO_AGENT_PAYLOAD_PREVIEW')=='1'
            if not cfg['args'].test or not cfg['args'].eval or not (cfg['env'].get('jointPushThreeAgentPreview') or (two_preview and cfg['env']['numAgents']==2)):
                raise ValueError('Payload override requires explicit team-size or two-agent inference preview')
            if preview_size is None or preview_mass is None:
                raise ValueError('Provide both preview size and mass')
            size=tuple(float(x) for x in preview_size.split(','));mass=float(preview_mass)
            if len(size)!=3 or any(not math.isfinite(x) or x<=0 for x in size) or size[2]<=.8 or not math.isfinite(mass) or mass<=0:
                raise ValueError('Invalid preview size/mass')
            self._shape_grid=[(size,mass)]
            self._preview_payload_override=True
        if os.environ.get('JOINT_PUSH_VIEW_MASS') is not None:
            raise ValueError('Random shape is separate from mass/three-agent diagnostic modes')
        if cfg['env']['jointPush']['goal_mode']!='box_forward' and cfg['env']['jointPush']['reward'].get('contact_mode') not in ('directional_surface','assigned_faces'):
            raise ValueError('Random shape currently uses aligned straight Push')
        super().__init__(cfg,*args,**kwargs)
        if os.environ.get('JOINT_PUSH_SHAPE_AUDIT_PATH'):
            self._audit_shapes(os.environ['JOINT_PUSH_SHAPE_AUDIT_PATH'])

    def _load_box_asset(self):
        super()._load_box_asset()
        self._shape_assets=[]
        for size,mass in self._shape_grid:
            options=gymapi.AssetOptions()
            options.density=mass/math.prod(size)
            options.angular_damping=.01;options.linear_damping=.01
            self._shape_assets.append(self.gym.create_box(self.sim,*size,options))
        ids=torch.randint(len(self._shape_grid),(self.num_envs,),device=self.device)
        self._shape_env_assignments=ids.cpu().tolist()
        self._box_size[:,2]=torch.tensor([size for size,mass in self._shape_grid],device=self.device)[ids]
        self._payload_masses=torch.tensor([mass for size,mass in self._shape_grid],device=self.device)[ids]

    def _build_box(self,env_id,object_id,env_ptr):
        if object_id!=2:return super()._build_box(env_id,object_id,env_ptr)
        original=self._push_asset
        try:
            self._push_asset=self._shape_assets[self._shape_env_assignments[env_id]]
            return super()._build_box(env_id,object_id,env_ptr)
        finally:self._push_asset=original

    def _audit_shapes(self,path):
        import json
        from pathlib import Path
        self._reset_envs(torch.arange(self.num_envs,device=self.device))
        sizes=self._box_size[:,2]
        assert (sizes[:,0]>=2).all() and (sizes[:,1]>=1).all() and (sizes[:,2]>=1).all()
        assert (self._payload_masses>=self._shape_spec['mass_min']).all() and (self._payload_masses<=max(m for _,m in self._shape_grid)).all()
        torch.testing.assert_close(self._box_bps[:,2].abs().amax(1),sizes/2)
        before_sizes=sizes.clone();before_masses=self._payload_masses.clone()
        max_mass_error=0.;max_inertia_error=0.
        for i,env in enumerate(self.envs):
            prop=self.gym.get_actor_rigid_body_properties(env,self._box_handles[4*i+2])[0]
            mass=float(self._payload_masses[i]);x,y,z=sizes[i].tolist()
            expected=[mass*(y*y+z*z)/12,mass*(x*x+z*z)/12,mass*(x*x+y*y)/12]
            measured=[prop.inertia.x.x,prop.inertia.y.y,prop.inertia.z.z]
            max_mass_error=max(max_mass_error,abs(prop.mass-mass))
            max_inertia_error=max(max_inertia_error,max(abs(a-b) for a,b in zip(expected,measured)))
        assert max_mass_error<.001 and max_inertia_error<.001
        self._reset_envs(torch.arange(min(3,self.num_envs),device=self.device))
        assert torch.equal(before_sizes,self._box_size[:,2]) and torch.equal(before_masses,self._payload_masses)
        torch.testing.assert_close(self._box_states[:,2,2],sizes[:,2]/2)
        assert torch.isfinite(self.obs_buf).all()
        report=dict(num_envs=self.num_envs,shape_min=sizes.amin(0).tolist(),shape_max=sizes.amax(0).tolist(),
            mass_min=float(self._payload_masses.min()),mass_max=float(self._payload_masses.max()),
            grid_combinations=len(self._shape_grid),unique_assigned=len(set(self._shape_env_assignments)),
            max_actual_mass_error=max_mass_error,max_actual_inertia_error=max_inertia_error,
            bbox_matches_physical_shape=True,partial_reset_preserves_shape_mass=True)
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2)+'\n');print('[JointPush random shape audit]',report,flush=True)
