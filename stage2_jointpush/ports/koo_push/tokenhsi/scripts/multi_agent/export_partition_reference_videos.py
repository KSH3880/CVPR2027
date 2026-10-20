"""Replay the actual Partition Push RSI/AMP references in Isaac Gym and export MP4."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import isaacgym  # Must precede torch.
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from utils.motion_lib import MotionLib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max-clips', type=int, default=0, help='Limit clips per family for a short preview')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames per clip for a short preview')
    args = parser.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.env_config).resolve()
    env = yaml.safe_load(cfg_path.read_text())['env']
    assert env['skillInitProb'] == [1., 0., 0.] and env['partitionPush']['rsi'] == 'loco_only'
    manifest = Path(env['motion_file']).resolve()
    entries = yaml.safe_load(manifest.read_text())['motions']
    ffmpeg = 'ffmpeg'
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.,0.,0.)
    params.dt = 1./args.fps  # No simulate() calls: this is a kinematic replay.
    params.physx.num_threads = 2
    sim = gym.create_sim(0,0,gymapi.SIM_PHYSX,params)
    if sim is None: raise RuntimeError('Isaac Gym simulation creation failed')
    plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0,0,1)
    gym.add_ground(sim,plane)
    scene = gym.create_env(sim,gymapi.Vec3(-20,-20,0),gymapi.Vec3(20,20,20),1)
    options = gymapi.AssetOptions(); options.disable_gravity = True
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_path = Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
    asset = gym.load_asset(sim,str(asset_path.parent),asset_path.name,options)
    actor = gym.create_actor(scene,asset,gymapi.Transform(),'reference',0,-1)
    names = gym.get_asset_rigid_body_names(asset)
    for body in range(len(names)):
        gym.set_rigid_body_color(scene,actor,body,gymapi.MESH_VISUAL,gymapi.Vec3(.15,.48,.85))
    props = gymapi.CameraProperties()
    props.width,props.height = args.width,args.height
    props.horizontal_fov = 45.
    camera = gym.create_camera_sensor(scene,props)
    gym.prepare_sim(sim)
    root_states=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_states=gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    keys = [names.index(name) for name in env['keyBodies']]
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path,28)
    small = ImageFont.truetype(font_path,22)
    report = dict(env_config=str(cfg_path),env_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        rsi_skill_probabilities=env['skillInitProb'],amp_expert_mapping=env['ampExpertMapping'],
        fps=args.fps,resolution=[args.width,args.height],kinematic_replay=True,
        note='Full source clips, not policy rollouts or frequency-weighted sampling; no object trajectory in manifest.',clips=[])
    all_clips=[]; by_family={}
    try:
        for family in ('loco','push'):
            lib = MotionLib(str(manifest),family,[1,2,3,4,6,7,9,10,11,12,13,14],
                [0,3,6,9,10,13,14,17,20,23,26,29,32],keys,'cpu')
            family_clips=[]
            for mid,row in enumerate(entries[family]):
                if args.max_clips and mid >= args.max_clips: break
                source=(manifest.parent/row['file']).resolve()
                raw=np.load(source,allow_pickle=True).item()
                assert raw['skeleton_tree']['node_names']==names
                duration=float(lib._motion_lengths[mid])
                frames=int(round(duration*args.fps))+1
                if args.max_frames: frames=min(frames,args.max_frames)
                title=source.parts[-3]
                filename=f'{family}_{mid+1:02d}.mp4'; dest=out/filename
                command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{args.width}x{args.height}','-r',str(args.fps),'-i','-','-an',
                    '-c:v','libopenh264','-b:v','12M','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
                encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
                anchor=lib.get_motion_state(torch.tensor([mid]),torch.tensor([0.]))[0][0,:2].numpy().copy()
                max_pose_error=0.
                max_dof_error=0.
                try:
                    for index in range(frames):
                        time=min(index/args.fps,duration)
                        mids=torch.tensor([mid]); times=torch.tensor([time],dtype=torch.float32)
                        root,rot,dof,*_=lib.get_motion_state(mids,times)
                        xyz=root[0].numpy().copy(); xyz[:2]-=anchor
                        root_states.zero_();root_states[0,:3]=torch.from_numpy(xyz)
                        root_states[0,3:7]=rot[0]
                        dof_states[:,0]=dof[0];dof_states[:,1]=0
                        gym.set_actor_root_state_tensor(sim,gymtorch.unwrap_tensor(root_states))
                        gym.set_dof_state_tensor(sim,gymtorch.unwrap_tensor(dof_states))
                        actual=gym.get_actor_rigid_body_states(scene,actor,gymapi.STATE_ALL)
                        actual_xyz=np.stack([actual['pose']['p'][a] for a in ('x','y','z')],-1)
                        error=float(np.linalg.norm(actual_xyz[0]-xyz))
                        assert np.isfinite(actual_xyz).all() and error<1.e-4, (family,mid,index,error)
                        actual_dof=gym.get_actor_dof_states(scene,actor,gymapi.STATE_ALL)['pos']
                        assert np.allclose(actual_dof,dof[0].numpy(),atol=1.e-5)
                        max_dof_error=max(max_dof_error,float(np.abs(actual_dof-dof[0].numpy()).max()))
                        max_pose_error=max(max_pose_error,error)
                        center=xyz.copy(); center[2]=.95
                        gym.set_camera_location(camera,scene,gymapi.Vec3(*(center+np.array([3.7,-5.2,1.7]))),gymapi.Vec3(*center))
                        gym.step_graphics(sim);gym.render_all_camera_sensors(sim)
                        pixels=gym.get_camera_image(sim,scene,camera,gymapi.IMAGE_COLOR)
                        image=Image.fromarray(np.asarray(pixels).reshape(args.height,args.width,4)[...,:3])
                        draw=ImageDraw.Draw(image)
                        draw.rectangle((0,0,args.width,112),fill=(20,24,31))
                        use='RSI + AMP locomotion' if family=='loco' else 'AMP Push reference'
                        draw.text((24,12),f'{use} | {mid+1}/{len(entries[family])} | {time:.2f} / {duration:.2f} s',font=font,fill='white')
                        draw.text((24,60),title,font=small,fill=(210,220,235))
                        if index==frames//2:image.save(out/f'{family}_{mid+1:02d}_preview.jpg',quality=95)
                        encoder.stdin.write(image.tobytes())
                finally:
                    encoder.stdin.close()
                if encoder.wait()!=0: raise RuntimeError('Video encoder failed: '+filename)
                record=dict(family=family,motion_id=mid,source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    weight=row['weight'],rsi_skipped_range=row.get('rsi_skipped_range'),frames=frames,duration_seconds=frames/args.fps,
                    max_root_position_error_m=max_pose_error,max_dof_error_rad=max_dof_error,video=filename)
                report['clips'].append(record)
                (out/'manifest.json').write_text(json.dumps(report,indent=2))
                family_clips.append(dest);all_clips.append(dest)
                print('EXPORTED',filename,frames,'frames',flush=True)
            by_family[family]=family_clips
        for name,clips in [('RSI_loco_all',by_family['loco']),('AMP_push_all',by_family['push']),('AMP_all',all_clips)]:
            listing=out/(name+'.txt')
            listing.write_text(''.join("file '"+str(p)+"'\n" for p in clips))
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),
                '-c','copy','-movflags','+faststart',str(out/(name+'.mp4'))],check=True)
        print('REFERENCE_EXPORT_COMPLETE',out,flush=True)
    finally:
        gym.destroy_sim(sim)


if __name__=='__main__':main()
