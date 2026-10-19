"""Frozen/reference replay of converted OMOMO humans and boxes. No policy."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['full','lift'],default='lift')
    parser.add_argument('--play',action='store_true')
    parser.add_argument('--phase',type=float,default=0.,help='Initial normalized reference time [0,1]')
    parser.add_argument('--frames',type=int,default=0,help='Exit after N rendered frames, 0 keeps viewer open')
    parser.add_argument('--output',default='output/omomo_lift_preview')
    args=parser.parse_args()
    gym=gymapi.acquire_gym()
    params=gymapi.SimParams()
    params.up_axis=gymapi.UP_AXIS_Z
    params.gravity=gymapi.Vec3(0,0,0)
    params.dt=1/30
    params.use_gpu_pipeline=False
    params.physx.use_gpu=False
    graphics=int(os.environ.get('TOKENHSI_GUI_GPU_INDEX','0'))
    sim=gym.create_sim(0,graphics,gymapi.SIM_PHYSX,params)
    if sim is None:raise RuntimeError('Failed to create simulator')
    viewer=None
    try:
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);gym.add_ground(sim,plane)
        opts=gymapi.AssetOptions();opts.disable_gravity=True;opts.fix_base_link=False
        asset=gym.load_asset(sim,str(ROOT/'tokenhsi/data/assets/mjcf'),'phys_humanoid_v3.xml',opts)
        body_names=gym.get_asset_rigid_body_names(asset)
        records=json.loads((ROOT/'tokenhsi/data/dataset_omomo_lift/conversion_summary.json').read_text())
        scenes=[]
        for r in records:
            if r['variant']!=args.variant:continue
            path=ROOT/'tokenhsi/data/dataset_omomo_lift/motions'/r['sequence']/args.variant/'phys_humanoid_v3'
            motion=SkeletonMotion.from_file(str(path/'ref_motion.npy'))
            box=np.load(path/'box_motion.npy');scale=np.load(path/'object_scale.npy')
            env=gym.create_env(sim,gymapi.Vec3(-1.5,-1.5,0),gymapi.Vec3(1.5,1.5,3),3)
            human=gym.create_actor(env,asset,gymapi.Transform(),r['sequence'],len(scenes),0)
            size=np.asarray(r['object_local_extents_raw'])
            object_asset=gym.create_box(sim,*[float(x) for x in size],opts)
            obj=gym.create_actor(env,object_asset,gymapi.Transform(),'box',len(scenes),0)
            gym.set_rigid_body_color(env,obj,0,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.85,.65,.25))
            colour=[gymapi.Vec3(.3,.65,.95),gymapi.Vec3(.4,.8,.5),gymapi.Vec3(.85,.45,.65)][len(scenes)]
            for j in range(len(body_names)):gym.set_rigid_body_color(env,human,j,gymapi.MESH_VISUAL_AND_COLLISION,colour)
            from tokenhsi.scripts.multi_agent.convert_bones_amp import DOF_BODIES, DOF_OFFSETS
            rv=Rotation.from_quat(motion.local_rotation.numpy().reshape(-1,4)).as_rotvec().reshape(len(box),15,3)
            dof=np.concatenate([rv[:,j] if hi-lo==3 else rv[:,j,1:2]
                for j,lo,hi in zip(DOF_BODIES,DOF_OFFSETS[:-1],DOF_OFFSETS[1:])],-1)
            order=[motion.skeleton_tree.index(n) for n in body_names]
            pos=motion.global_translation.numpy()[:,order].copy()
            rot=motion.global_rotation.numpy()[:,order].copy()
            xy=motion.root_translation[0,:2].numpy().copy()
            pos[:,:,:2]-=xy;box[:,:2]-=xy
            scenes.append(dict(env=env,human=human,box=obj,pos=pos,rot=rot,track=box,scale=scale,dof=dof,report=r))
        gym.prepare_sim(sim)
        gym.simulate(sim);gym.fetch_results(sim,True)
        root_tensor=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        dof_tensor=gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
        viewer=gym.create_viewer(sim,gymapi.CameraProperties())
        if viewer is None:raise RuntimeError('Failed to create viewer')
        bindings=[(gymapi.KEY_SPACE,'PLAY'),(gymapi.KEY_R,'RANDOM'),(gymapi.KEY_RIGHT,'NEXT'),
                  (gymapi.KEY_LEFT,'PREV'),(gymapi.KEY_N,'SCENE'),(gymapi.KEY_HOME,'HOME'),(gymapi.KEY_S,'SAVE')]
        for key,action in bindings:gym.subscribe_viewer_keyboard_event(viewer,key,action)
        output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
        phase=float(np.clip(args.phase,0,1));playing=args.play;selected=-1;count=0
        def camera():
            env=scenes[max(selected,0)]['env']
            if selected<0:
                gym.viewer_camera_look_at(viewer,env,gymapi.Vec3(4.5,-8,3.8),gymapi.Vec3(3,0,1.1))
            else:gym.viewer_camera_look_at(viewer,env,gymapi.Vec3(-2.8,-3.2,2.4),gymapi.Vec3(0,0,1.1))
        def label():
            print('[OMOMO preview]',args.variant,'playing=',playing,'phase=',round(phase,3),flush=True)
            for i,s in enumerate(scenes):
                frame=min(int(phase*(len(s['pos'])-1)),len(s['pos'])-1)
                print(i+1,s['report']['sequence'],'frame',frame,'source frame',s['report']['source_frames_inclusive'][0]+frame,flush=True)
        camera();label()
        print('Space play/pause | R random frozen poses | arrows frame | N overview/clip camera | Home start | S screenshot',flush=True)
        previous=time.monotonic()
        while not gym.query_viewer_has_closed(viewer):
            now=time.monotonic();dt=min(now-previous,.1);previous=now
            save=False
            for event in gym.query_viewer_action_events(viewer):
                if event.value<=0:continue
                if event.action=='PLAY':playing=not playing
                elif event.action=='RANDOM':phase=float(np.random.random());playing=False
                elif event.action=='HOME':phase=0.;playing=False
                elif event.action=='NEXT':phase=min(1.,phase+1/(len(scenes[max(selected,0)]['pos'])-1));playing=False
                elif event.action=='PREV':phase=max(0.,phase-1/(len(scenes[max(selected,0)]['pos'])-1));playing=False
                elif event.action=='SCENE':selected=(selected+2)%(len(scenes)+1)-1;camera()
                elif event.action=='SAVE':save=True
                label()
            if playing:phase=(phase+dt/((len(scenes[max(selected,0)]['pos'])-1)/30))%1.
            # Apply RSI-style root/DOF states, with no physics stepping during replay.
            root_tensor.zero_();dof_tensor.zero_()
            for s in scenes:
                frame=min(int(round(phase*(len(s['pos'])-1))),len(s['pos'])-1)
                offset=np.zeros(3)  # Isaac Gym state setters use env-local poses.
                h=gym.get_actor_index(s['env'],s['human'],gymapi.DOMAIN_SIM)
                o=gym.get_actor_index(s['env'],s['box'],gymapi.DOMAIN_SIM)
                root_tensor[h,:3]=torch.tensor(s['pos'][frame,0]+offset)
                root_tensor[h,3:7]=torch.tensor(s['rot'][frame,0])
                root_tensor[o,:3]=torch.tensor(s['track'][frame,:3]+offset)
                root_tensor[o,3:7]=torch.tensor(s['track'][frame,3:])
                first=gym.get_actor_dof_index(s['env'],s['human'],0,gymapi.DOMAIN_SIM)
                dof_tensor[first:first+32,0]=torch.tensor(s['dof'][frame])
                gym.set_actor_scale(s['env'],s['box'],float(s['scale'][frame]))
            gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root_tensor))
            gym.set_dof_state_tensor(sim,gymtorch.unwrap_tensor(dof_tensor))
            gym.step_graphics(sim);gym.draw_viewer(viewer,sim,True)
            if count==0:
                for s in scenes:
                    origin=gym.get_env_origin(s['env'])
                    print('[scene origin]',s['report']['sequence'],origin,flush=True)
            if count==0 or save:gym.write_viewer_image_to_file(viewer,str(output/f'view_{count:06d}.png'))
            count+=1
            if args.frames and count>=args.frames:break
            time.sleep(max(0,1/30-(time.monotonic()-now)))
    finally:
        if viewer is not None:gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

if __name__=='__main__':main()
