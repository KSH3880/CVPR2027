"""Design and inspect an underhand pose; CPU PhysX, GPU 6 graphics only.

The box is dynamic in probes. Frozen viewing is geometry only. Anchored-root
and free-root probes are reported separately, neither is a learned lift policy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from isaacgym import gymapi, gymtorch, gymutil  # before torch
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

XML=ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'

def design(height):
    xml=ET.parse(XML)
    bodies=list(xml.find('worldbody').find('body').iter('body'))
    names=[b.get('name') for b in bodies]
    # Asset body order has no hidden joint bodies; local rotations are native expmaps.
    parent={c.get('name'):p.get('name') for p in bodies for c in p.findall('body')}
    offsets={b.get('name'):np.fromstring(b.get('pos','0 0 0'),sep=' ') for b in bodies}
    joints=[(b.get('name'),j) for b in bodies for j in b.findall('joint')]
    lo=np.deg2rad([float(j.get('range').split()[0]) for _,j in joints])
    hi=np.deg2rad([float(j.get('range').split()[1]) for _,j in joints])
    slices={};k=0
    for b in bodies:
        n=len(b.findall('joint'));slices[b.get('name')]=slice(k,k+n);k+=n
    # Stand feet at mesh floor, without changing the asset.
    feet=[]
    for side in ('right','left'):
        mesh=XML.parent/'../smpl_foot_meshes'/f'{side}_foot.obj'
        verts=np.array([[float(x) for x in l.split()[1:4]] for l in mesh.read_text().splitlines() if l.startswith('v ')])
        foot=offsets[f'{side}_thigh']+offsets[f'{side}_shin']+offsets[f'{side}_foot']
        feet.append(foot[2]+verts[:,2].min())
    root_z=-min(feet)+.002
    def fk(q):
        positions={'pelvis':np.array([0.,0.,root_z])};rotations={'pelvis':np.eye(3)}
        for b in bodies[1:]:
            name=b.get('name');par=parent[name]
            positions[name]=positions[par]+rotations[par]@offsets[name]
            v=q[slices[name]]
            local=Rotation.from_rotvec(v if len(v)==3 else ([0.,v[0],0.] if len(v)==1 else [0.,0.,0.])).as_matrix()
            rotations[name]=rotations[par]@local
        return positions,rotations
    arm_ids=np.r_[np.arange(slices['right_upper_arm'].start,slices['right_lower_arm'].stop),np.arange(slices['left_upper_arm'].start,slices['left_lower_arm'].stop)]
    gap=.258947-.1875
    # Symmetric, slightly inward-facing horizontal forearms; box x is optimized.
    theta=np.deg2rad(35)
    def targets(cx):
        out=[]
        for sign in (-1,1):
            direction=np.array([np.cos(theta),-sign*np.sin(theta),0.])
            middle=np.array([cx,sign*.18,height-.35/2-.04])
            out.extend([middle+direction*gap/2,middle-direction*gap/2])
        return np.array(out)
    def unpack(x):
        q=np.zeros(32);q[arm_ids]=x[:8];return q
    def points(q):
        p,r=fk(q);out=[]
        for side in ('right','left'):
            e=p[f'{side}_lower_arm'];h=p[f'{side}_hand']
            out.extend([h,e+.1875/.258947*(h-e)])
        return np.array(out)
    def residual(x):
        return np.r_[(points(unpack(x))-targets(x[-1])).ravel()*50, .015*x[:8], .01*(x[-1]-.4)]
    bounds=(np.r_[lo[arm_ids]+1e-5,.30],np.r_[hi[arm_ids]-1e-5,.60])
    rng=np.random.default_rng(42);best=None
    for _ in range(16):
        initial=np.r_[rng.uniform(bounds[0][:8],bounds[1][:8]),.4]
        sol=least_squares(residual,initial,bounds=bounds,max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if best is None or np.linalg.norm(sol.fun)<np.linalg.norm(best.fun):best=sol
    q=unpack(best.x);p,r=fk(q)
    return dict(height=height,root_z=root_z,box_x=float(best.x[-1]),dof=q.tolist(),
                body_positions={n:v.tolist() for n,v in p.items()},
                targets=targets(best.x[-1]).tolist(),points=points(q).tolist(),
                max_target_error_m=float(np.linalg.norm(points(q)-targets(best.x[-1]),axis=-1).max()),
                joint_names=[j.get('name') for _,j in joints],
                joint_degrees=np.rad2deg(q).tolist(),joint_limits_valid=bool(((q>=lo)&(q<=hi)).all()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height',type=float,default=1.7)
    parser.add_argument('--output',default=str(ROOT.parent/'runs/lift_support_pose_170'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pose=design(args.height)
    (out/'design.json').write_text(json.dumps(pose,indent=2)+'\n')
    print('[DESIGN]',json.dumps({k:pose[k] for k in ('root_z','box_x','max_target_error_m','joint_limits_valid')}),flush=True)
    gym=gymapi.acquire_gym();params=gymapi.SimParams();params.up_axis=gymapi.UP_AXIS_Z
    params.dt=1/120;params.substeps=2;params.gravity=gymapi.Vec3(0,0,-9.81)
    params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=2
    params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
    params.physx.contact_offset=.002;params.physx.rest_offset=0
    graphics=-1 if args.headless else int(os.environ.get('TOKENHSI_GUI_GPU_INDEX','6'))
    if graphics not in (-1,6):raise RuntimeError('Only physical GPU 6 allowed')
    sim=gym.create_sim(0,graphics,gymapi.SIM_PHYSX,params)
    if sim is None:raise RuntimeError('create_sim failed')
    viewer=None
    try:
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=1.;plane.dynamic_friction=1.;gym.add_ground(sim,plane)
        scenes=[]
        for anchored in (True,False):
            opts=gymapi.AssetOptions();opts.fix_base_link=anchored;opts.default_dof_drive_mode=int(gymapi.DOF_MODE_POS)
            asset=gym.load_asset(sim,str(XML.parent),XML.name,opts)
            env=gym.create_env(sim,gymapi.Vec3(-1.5,-1.5,0),gymapi.Vec3(1.5,1.5,3),2)
            t=gymapi.Transform();t.p=gymapi.Vec3(0,0,pose['root_z'])
            human=gym.create_actor(env,asset,t,'anchored' if anchored else 'free',len(scenes),0)
            props=gym.get_actor_dof_properties(env,human)
            props['driveMode'].fill(gymapi.DOF_MODE_POS)
            gym.set_actor_dof_properties(env,human,props)
            initial_states=np.zeros(32,dtype=gymapi.DofState.dtype);initial_states['pos']=np.array(pose['dof'],dtype=np.float32)
            gym.set_actor_dof_states(env,human,initial_states,gymapi.STATE_ALL)
            gym.set_actor_dof_position_targets(env,human,np.array(pose['dof'],dtype=np.float32))
            for b in range(gym.get_asset_rigid_body_count(asset)):
                gym.set_rigid_body_color(env,human,b,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.3,.65,.95) if anchored else gymapi.Vec3(.4,.8,.5))
            bo=gymapi.AssetOptions();bo.density=5/(.55*.65*.35);bo.angular_damping=.01;bo.linear_damping=.01
            ba=gym.create_box(sim,.55,.65,.35,bo)
            t=gymapi.Transform();t.p=gymapi.Vec3(pose['box_x'],0,args.height)
            box=gym.create_actor(env,ba,t,'box',len(scenes),0)
            gym.set_rigid_body_color(env,box,0,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.9,.65,.25))
            for actor in (human,box):
                shapes=gym.get_actor_rigid_shape_properties(env,actor)
                for shape in shapes:shape.friction=1.;shape.restitution=0
                gym.set_actor_rigid_shape_properties(env,actor,shapes)
            assert gym.get_asset_dof_names(asset)==pose['joint_names']
            ids={n:gym.find_actor_rigid_body_handle(env,human,n) for n in gym.get_asset_rigid_body_names(asset)}
            scenes.append(dict(env=env,human=human,box=box,anchored=anchored,ids=ids))
        gym.prepare_sim(sim)
        gym.simulate(sim);gym.fetch_results(sim,True)
        roots=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        dofs=gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
        rigid=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        contacts=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        q=torch.tensor(pose['dof'],dtype=torch.float32)
        initial_root=roots.clone();initial_dof=dofs.clone()
        for i,s in enumerate(scenes):
            h=gym.get_actor_index(s['env'],s['human'],gymapi.DOMAIN_SIM)
            b=gym.get_actor_index(s['env'],s['box'],gymapi.DOMAIN_SIM)
            initial_root[h,:3]=torch.tensor([0,0,pose['root_z']]);initial_root[h,3:7]=torch.tensor([0,0,0,1.]);initial_root[h,7:]=0
            initial_root[b,:3]=torch.tensor([pose['box_x'],0,args.height]);initial_root[b,3:7]=torch.tensor([0,0,0,1.]);initial_root[b,7:]=0
            first=gym.get_actor_dof_index(s['env'],s['human'],0,gymapi.DOMAIN_SIM)
            initial_dof[first:first+32,0]=q;initial_dof[first:first+32,1]=0
            s.update(root=h,box_root=b,first=first)
        position_targets=initial_dof[:,0].contiguous()  # Gym retains this buffer; keep it alive.
        def refresh():
            gym.refresh_actor_root_state_tensor(sim);gym.refresh_dof_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
        def reset():
            roots.copy_(initial_root);dofs.copy_(initial_dof)
            ids=torch.tensor([s['box_root'] for s in scenes]+[s['root'] for s in scenes if not s['anchored']],dtype=torch.int32)
            gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(roots),gymtorch.unwrap_tensor(ids),len(ids))
            gym.set_dof_state_tensor(sim,gymtorch.unwrap_tensor(dofs))
            gym.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(position_targets))
        def step():
            gym.simulate(sim);gym.fetch_results(sim,True);refresh()
        def metrics(s):
            state=gym.get_actor_rigid_body_states(s['env'],s['human'],gymapi.STATE_POS)
            pp=np.array([[state['pose']['p'][s['ids'][n]][axis] for axis in ('x','y','z')] for n in ('right_hand','left_hand','right_lower_arm','left_lower_arm')])
            expected=np.array([pose['body_positions'][n] for n in ('right_hand','left_hand','right_lower_arm','left_lower_arm')])
            bs=roots[s['box_root']].numpy().copy()
            quat=bs[3:7];angle=float(np.rad2deg(Rotation.from_quat(quat).magnitude())) if np.linalg.norm(quat)>0 else float('nan')
            force={n:contacts[gym.get_actor_rigid_body_index(s['env'],s['human'],index,gymapi.DOMAIN_SIM)].numpy().copy() for n,index in s['ids'].items()}
            return dict(anchored_root=s['anchored'],box_center=bs[:3].tolist(),box_angle_deg=angle,root_z=float(roots[s['root'],2]),
                        hand_elbow_world=pp.tolist(),fk_max_error_m=float(np.linalg.norm(pp-expected,axis=-1).max()),
                        hand_forearm_contact_forces_N={n:force[n].tolist() for n in ('right_hand','left_hand','right_lower_arm','left_lower_arm')})
        reset();step()
        init=[metrics(s) for s in scenes]
        report=dict(design=pose,initial_sim=init,probe_description='Unattached 5kg box under gravity; ordinary asset position PD. Blue root anchored / green root free. Not a lift trajectory or learned policy.')
        if args.probe:
            history=[]
            for i in range(360):
                step()
                if (i+1)%30==0:history.append(dict(time_s=(i+1)/120,scenes=[metrics(s) for s in scenes]))
            report['probe_history']=history
            report['probe_final']=[metrics(s) for s in scenes]
            print('[PROBE]',json.dumps(report['probe_final']),flush=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if args.headless:return
        viewer=gym.create_viewer(sim,gymapi.CameraProperties())
        if viewer is None:raise RuntimeError('create_viewer failed')
        for key,action in ((gymapi.KEY_SPACE,'PLAY'),(gymapi.KEY_R,'RESET'),(gymapi.KEY_N,'CAMERA'),(gymapi.KEY_S,'SAVE')):gym.subscribe_viewer_keyboard_event(viewer,key,action)
        selected=1;playing=False;frame=0
        def camera():
            env=scenes[selected]['env']
            gym.viewer_camera_look_at(viewer,env,gymapi.Vec3(2.5,-3,2.4),gymapi.Vec3(.15,0,1.25))
        def draw():
            gym.clear_lines(viewer)
            if not playing:
                for s in scenes:
                    for point in pose['points']:
                        t=gymapi.Transform();t.p=gymapi.Vec3(*point)
                        sphere=gymutil.WireframeSphereGeometry(.013,8,8,None,color=(1.,.1,.1))
                        gymutil.draw_lines(sphere,gym,viewer,s['env'],t)
            gym.step_graphics(sim);gym.draw_viewer(viewer,sim,True)
        reset();step();camera();draw()
        gym.write_viewer_image_to_file(viewer,str(out/'pose.png'))
        print('[VIEW] Box center=%.3fm; Space gravity play/pause, R reset, N anchored/free camera, S screenshot'%args.height,flush=True)
        while not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value<=0:continue
                if event.action=='PLAY':playing=not playing
                elif event.action=='RESET':playing=False;reset();step()
                elif event.action=='CAMERA':selected=1-selected;camera()
                elif event.action=='SAVE':gym.write_viewer_image_to_file(viewer,str(out/f'view_{frame:06d}.png'))
            if playing:
                for _ in range(4):step()
            draw();frame+=1
            if args.frames and frame>=args.frames:break
            time.sleep(1/30)
    finally:
        if viewer is not None:gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

if __name__=='__main__':main()
