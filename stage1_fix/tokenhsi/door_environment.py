"""Door physics fixture and contact/self-closing check; no policy or training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from isaacgym import gymapi,gymtorch
import argparse
from dataclasses import asdict
import json
import math
import os
import numpy as np
import torch
from tokenhsi.utils.door_asset import DoorSpec
from tokenhsi.env.tasks.multi_agent.door_scene import DoorFixture


class DoorEnvironment:
    def __init__(self,num_envs=4,headless=True,spec=DoorSpec(),output=None):
        self.spec,self.num_envs,self.headless=spec,num_envs,headless
        self.output=Path(output or ROOT/'output/door_environment_check')
        self.output.mkdir(parents=True,exist_ok=True)
        self.gym=gymapi.acquire_gym()
        params=gymapi.SimParams()
        params.dt=1/60;params.substeps=2;params.up_axis=gymapi.UP_AXIS_Z
        params.gravity=gymapi.Vec3(0,0,-9.81)
        params.use_gpu_pipeline=True
        params.physx.use_gpu=True
        params.physx.num_position_iterations=6
        params.physx.num_velocity_iterations=2
        params.physx.contact_offset=.002
        params.physx.rest_offset=0.
        params.physx.num_threads=4
        self.dt=params.dt
        self.sim=self.gym.create_sim(0,-1 if headless else 0,
                                     gymapi.SIM_PHYSX,params)
        if self.sim is None:raise RuntimeError('Could not create door physics simulation')
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1)
        self.gym.add_ground(self.sim,plane)
        self.doors=DoorFixture(self.gym,self.sim,spec)
        probe_options=gymapi.AssetOptions()
        probe_options.disable_gravity=True
        probe_options.density=1000
        probe_options.linear_damping=.5
        probe=self.gym.create_sphere(self.sim,.045,probe_options)
        self.envs=[];self.probe_actors=[];self.probe_bodies=[]
        for i in range(num_envs):
            env=self.gym.create_env(self.sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,3),max(1,int(math.sqrt(num_envs))))
            pose=gymapi.Transform()
            pose.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1),(i%4)*math.pi/2)
            self.doors.add(env,i,pose)
            probe_pose=gymapi.Transform();probe_pose.p=gymapi.Vec3(-1,-1,.3)
            actor=self.gym.create_actor(env,probe,probe_pose,'contact_probe',i,0)
            self.gym.set_rigid_body_color(env,actor,0,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.1,.65,1.))
            self.probe_actors.append(self.gym.get_actor_index(env,actor,gymapi.DOMAIN_SIM))
            self.probe_bodies.append(self.gym.get_actor_rigid_body_index(env,actor,0,gymapi.DOMAIN_SIM))
            self.envs.append(env)
        self.gym.prepare_sim(self.sim)
        self.doors.bind_tensors()
        self.roots=gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.contacts=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.device=self.doors.dof_states.device
        self.probe_actors=torch.tensor(self.probe_actors,device=self.device,dtype=torch.int32)
        self.probe_bodies=torch.tensor(self.probe_bodies,device=self.device,dtype=torch.long)
        self.forces=torch.zeros((self.gym.get_sim_rigid_body_count(self.sim),3),device=self.device)
        self.viewer=None
        if not headless:
            camera=gymapi.CameraProperties();camera.width=1280;camera.height=900
            self.viewer=self.gym.create_viewer(self.sim,camera)
            if self.viewer is None:raise RuntimeError('Could not create door viewer')
            self.gym.viewer_camera_look_at(self.viewer,self.envs[0],gymapi.Vec3(-3,-1.4,1.8),gymapi.Vec3(0,0,1.05))
        self.step()

    def step(self):
        self.gym.simulate(self.sim);self.gym.fetch_results(self.sim,True)
        self.doors.refresh();self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):raise KeyboardInterrupt
            self.gym.step_graphics(self.sim);self.gym.draw_viewer(self.viewer,self.sim,True)
            self.gym.sync_frame_time(self.sim)
        state=self.doors.observe()
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise RuntimeError('Nonfinite door physics state')
        return state

    def reset_probes(self,positions):
        states=self.roots[self.probe_actors.long()].clone()
        states[:,:3]=torch.as_tensor(positions,device=self.device,dtype=torch.float32)
        states[:,3:7]=torch.tensor([0,0,0,1],device=self.device)
        states[:,7:]=0.
        self.roots[self.probe_actors.long()]=states
        if not self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.roots),
                    gymtorch.unwrap_tensor(self.probe_actors),len(self.probe_actors)):
            raise RuntimeError('Probe reset failed')

    def handle_targets(self,angles,clearance=0.):
        state=self.doors.observe()
        base=state['base_pose'].cpu().numpy()
        result=[]
        for angle,pose in zip(angles,base):
            x,y,z,w=pose[3:7]
            yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            pos=self.spec.handle_position(float(angle),pose[:3],yaw)
            pos-=clearance*np.array([math.cos(yaw+angle),math.sin(yaw+angle),0])
            result.append(pos)
        return np.asarray(result)

    def park_probes(self):
        self.reset_probes(self.handle_targets(np.zeros(self.num_envs),clearance=1.))

    def push(self,steps=180):
        self.doors.reset();self.park_probes();self.step()
        self.reset_probes(self.handle_targets(np.zeros(self.num_envs),clearance=.10))
        max_angles=torch.zeros(self.num_envs,device=self.device)
        peak_contact=torch.zeros_like(max_angles)
        for _ in range(steps):
            state=self.doors.observe()
            target_angles=np.minimum(state['angle'].cpu().numpy()+.18,math.radians(95))
            targets=torch.tensor(self.handle_targets(target_angles,clearance=.035),device=self.device,dtype=torch.float32)
            probe_state=self.roots[self.probe_actors.long()]
            force=120*(targets-probe_state[:,:3])-8*probe_state[:,7:10]
            force=force*(18/force.norm(dim=-1,keepdim=True).clamp_min(18))
            self.forces.zero_();self.forces[self.probe_bodies]=force
            self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(self.forces),None,gymapi.ENV_SPACE)
            state=self.step()
            max_angles=torch.maximum(max_angles,state['angle'])
            peak_contact=torch.maximum(peak_contact,self.contacts[self.probe_bodies].norm(dim=-1))
        return max_angles.cpu().tolist(),peak_contact.cpu().tolist()

    def snapshot(self,name):
        if self.viewer:
            self.gym.write_viewer_image_to_file(self.viewer,str(self.output/(name+'.png')))

    def validate(self):
        report=dict(spec=asdict(self.spec),num_envs=self.num_envs,training=False,checks={})
        self.doors.reset();self.park_probes()
        for _ in range(30):self.step()
        assert self.doors.observe()['angle'].abs().max()<.01
        self.snapshot('closed')
        # Geometry check at closed/half/full opening, including rotated instances.
        geometry_error=0.
        for angle in [0.,math.pi/4,math.pi/2]:
            self.doors.reset(angle=angle);self.step()
            state=self.doors.observe()
            expected=torch.tensor(self.handle_targets(state['angle'].cpu().numpy()),device=self.device,dtype=torch.float32)
            geometry_error=max(geometry_error,float((expected-state['handle_position']).norm(dim=-1).max()))
        assert geometry_error<1e-4,geometry_error
        # A partial reset must preserve every unselected door.
        self.doors.reset(angle=math.pi/3);self.step()
        before=self.doors.observe()['angle'].clone()
        self.doors.reset(door_ids=[0]);after=self.doors.observe()['angle']
        assert abs(float(after[0]))<1e-6
        torch.testing.assert_close(after[1:],before[1:])
        opening,contact=self.push()
        self.snapshot('pushed_open')
        assert min(opening)>math.radians(20),opening
        assert min(contact)>.1,contact
        # Remove the contacting body and observe the closing spring, not a reset.
        self.park_probes()
        released=self.doors.observe()['angle'].clone()
        for _ in range(720):self.step()
        returned=self.doors.observe()['angle']
        assert returned.abs().max()<math.radians(3),returned
        self.snapshot('released_closed')
        # At 90deg the compliant hinge should not slam closed in 0.2 seconds.
        self.doors.reset(angle=math.pi/2)
        for _ in range(12):self.step()
        early=self.doors.observe()['angle']
        assert early.min()>math.radians(85),early
        report['checks']=dict(geometry_max_error_m=geometry_error,partial_reset=True,
            pushed_open_degrees=np.rad2deg(opening).tolist(),peak_probe_contact_n=contact,
            released_degrees=torch.rad2deg(released).cpu().tolist(),
            after_release_12s_degrees=torch.rad2deg(returned).cpu().tolist(),
            after_release_0_2s_from_90_degrees=torch.rad2deg(early).cpu().tolist(),
            spring_torque_at_90_nm=abs(float(self.spec.closing_torque(math.pi/2))))
        self.doors.reset();self.step()
        (self.output/'physics_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return report

    def close(self):
        if self.viewer:self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=4,help='Physics check/viewer environments, not training')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'output/door_environment_check')
    parser.add_argument('--stiffness',type=float,default=6.)
    parser.add_argument('--damping',type=float,default=3.)
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args()
    if args.num_envs<1 or args.repeats<1:parser.error('Counts must be positive')
    env=DoorEnvironment(args.num_envs,args.headless,DoorSpec(spring_stiffness=args.stiffness,spring_damping=args.damping),args.output)
    try:
        for _ in range(args.repeats):env.validate()
    finally:env.close()
