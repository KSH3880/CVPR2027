"""Retarget cloned carry poses to physical 32-DOF arms with collision-aware IK.

Preserves lower-body articulated angles and 15cm interior grip locations.
Output is a reference-pose candidate, not a dynamically valid expert rollout.
Run validate_carry_physics.py separately before selecting RSI snapshots.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from carry_joint_preview import (DOF_BODIES, DOF_SIZES, SKILL_TAGS, SKILL_TITLES,
                                 forward_kinematics, pack, render)

ASSET=Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')


def sdf_box(points, size):
    d=np.abs(points)-size/2
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)


def physical_rotations(data):
    q=np.zeros_like(data['local_rotation_xyzw'])
    q[...,3]=1
    q[:,:,0]=data['root_rotation_xyzw']
    offset=0
    for j,size in zip(DOF_BODIES,DOF_SIZES):
        rv=np.zeros(q.shape[:2]+(3,))
        if size==3:
            rv[:]=data['dof_position'][:,:,offset:offset+3]
        else:
            rv[:,:,1]=data['dof_position'][:,:,offset]
        q[:,:,j]=R.from_rotvec(rv.reshape(-1,3)).as_quat().reshape(q.shape[:2]+(4,))
        offset+=size
    return q


def feet_floor_lift(q, roots, data):
    xyz,rot=forward_kinematics(q[:,0],roots[:,0],data['parent_indices'],data['local_translation'])
    floor=np.full(len(q),np.inf)
    for name in ('left_foot','right_foot'):
        j=data['joint_names'].tolist().index(name)
        path=ASSET.parent/'../smpl_foot_meshes'/ (name+'.obj')
        vertices=np.asarray([[float(v) for v in line.split()[1:4]] for line in path.read_text().splitlines()
                             if line.startswith('v ')])
        r=R.from_quat(rot[:,j])
        min_z=np.stack([r.apply(v)[:,2] for v in vertices],axis=1).min(1)+xyz[:,j,2]
        floor=np.minimum(floor,min_z)
    # Constant lift over the clip avoids injecting artificial vertical velocity.
    lift=max(0.,.001-float(floor.min()))
    return lift


def correct(data):
    q=physical_rotations(data)
    roots=data['root_position'].copy()
    lift=feet_floor_lift(q,roots,data)
    roots[:,:,2]+=lift
    xyz,world_q=forward_kinematics(q[:,0],roots[:,0],data['parent_indices'],data['local_translation'])
    inv=R.from_quat(data['box_rotation_xyzw']).inv()
    size=data['box_size']
    xml=ET.parse(ASSET).getroot()
    traces=[]
    for shoulder,elbow,hand in [(3,4,5),(6,7,8)]:
        shoulder_name=data['joint_names'][shoulder]
        elbow_name=data['joint_names'][elbow]
        body=next(b for b in xml.findall('.//body') if b.get('name')==shoulder_name)
        lower=next(b for b in xml.findall('.//body') if b.get('name')==elbow_name)
        joints=body.findall('joint')+lower.findall('joint')
        bounds=np.asarray([np.fromstring(j.get('range'),sep=' ') for j in joints])*np.pi/180
        upper_segment=np.fromstring(body.find('geom').get('fromto'),sep=' ').reshape(2,3)
        lower_segment=np.fromstring(lower.find('geom').get('fromto'),sep=' ').reshape(2,3)
        upper_points=np.linspace(*upper_segment,24)
        lower_points=np.linspace(*lower_segment,24)
        prev_delta=np.zeros(4)
        original=np.concatenate([R.from_quat(q[:,0,shoulder]).as_rotvec(),
                                 R.from_quat(q[:,0,elbow]).as_rotvec()[:,1:2]],axis=1)
        for f in range(len(q)):
            shoulder_position=inv[f].apply(xyz[f,shoulder]-data['box_position'][f])
            torso_rotation=inv[f]*R.from_quat(world_q[f,1])
            reference_hand=inv[f].apply(data['body_position'][f,0,hand]-data['box_position'][f])
            reference_hand[2]+=lift
            # Smoothly enable a side-face grasp as the source hands approach.
            gate=np.clip((size[1]/2+.15-abs(reference_hand[1]))/.15,0,1)
            gate*=np.clip((size[2]/2+.15-abs(reference_hand[2]))/.15,0,1)
            target=reference_hand.copy()
            desired_x=np.sign(reference_hand[0])*(size[0]/2+.04-.005)
            target[0]=(1-gate)*target[0]+gate*desired_x
            base=original[f]

            def locations(values):
                upper=torso_rotation*R.from_rotvec(values[:3])
                elbow_position=shoulder_position+upper.apply(data['local_translation'][elbow])
                lower_r=upper*R.from_rotvec([0,values[3],0])
                hand_position=elbow_position+lower_r.apply(data['local_translation'][hand])
                upper_cloud=shoulder_position+upper.apply(upper_points)
                lower_cloud=elbow_position+lower_r.apply(lower_points)
                return hand_position,upper_cloud,lower_cloud

            def residual(values):
                h,u,l=locations(values)
                collision=np.concatenate([np.maximum(.045+.003-sdf_box(u,size),0),
                                          np.maximum(.04+.003-sdf_box(l,size),0),
                                          np.maximum(.04-.008-sdf_box(h[None],size),0)])
                return np.concatenate([(h-target)/.003,collision/.002,
                                       (values-base)*.8,((values-base)-prev_delta)*.4])

            lo=np.maximum(bounds[:,0]+1e-5,base-.85)
            hi=np.minimum(bounds[:,1]-1e-5,base+.85)
            result=least_squares(residual,np.clip(base+prev_delta,lo+1e-7,hi-1e-7),
                                 bounds=(lo,hi),max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            q[f,0,shoulder]=R.from_rotvec(result.x[:3]).as_quat()
            q[f,0,elbow]=R.from_rotvec([0,result.x[3],0]).as_quat()
            prev_delta=result.x-base
            h,_,_=locations(result.x)
            traces.append(dict(frame=f,arm=str(shoulder_name),grasp_gate=float(gate),
                               target_error_m=float(np.linalg.norm(h-target)),
                               maximum_joint_change_deg=float(np.abs(prev_delta).max()*180/np.pi)))
    # Clone articulated pose exactly; the existing opposite root rotation stays.
    q[:,1,1:]=q[:,0,1:]
    output=pack(q,roots,data['parent_indices'],data['local_translation'],
                np.concatenate([data['box_position'],data['box_rotation_xyzw']],axis=-1),
                size,int(data['fps']),data['joint_names'].tolist())
    for key in ('skill','source_rsi_allowed','grip_inset_m'):
        output[key]=data[key]
    output['physics_valid_frame_mask']=np.zeros(len(q),dtype=bool)
    output['ik_applied']=np.asarray(True)
    output['simulator_pose_exact']=np.asarray(True)
    output['render_caption']=np.asarray('32-DOF arm IK | 15 cm interior grips | unchanged leg angles')
    output['render_footer']=np.asarray('Corrected reference | pending physics screen | not a policy rollout')
    unchanged=[0,1,2]+list(range(9,15))
    physical=physical_rotations(data)
    np.testing.assert_allclose(q[:,:,unchanged],physical[:,:,unchanged],atol=1e-12)
    return output,dict(root_height_lift_m=lift,arm_fit=traces,
                       maximum_arm_dof_change_deg=max(t['maximum_joint_change_deg'] for t in traces),
                       maximum_hand_target_error_m=max(t['target_error_m'] for t in traces))


def finalize(args):
    report=json.loads((args.audit/'report.json').read_text())
    for path,digest in report['input_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Physics report does not match current candidate: '+path)
    selected={}
    for skill in SKILL_TAGS:
        path=args.output/skill
        candidate=path/'cooperative_rsi_candidate.npz'
        if report['input_hashes'].get(str(candidate)) != hashlib.sha256(candidate.read_bytes()).hexdigest():
            raise ValueError('Requested candidate was not screened by this report: '+str(candidate))
        data=dict(np.load(path/'cooperative_rsi_candidate.npz',allow_pickle=False))
        mask=np.load(args.audit/skill/'cooperative/screen_masks.npz')['conservative_rsi_screen_pass'].copy()
        with (args.audit/skill/'cooperative/frames.csv').open() as stream:
            records=list(csv.DictReader(stream))
        if skill=='carryWith':
            # Carry snapshots additionally need both carriers to establish a
            # hand contact in at least half the six simulated steps.
            mask &= np.asarray([float(r['all_carriers_hand_contact_step_fraction'])>=.5 for r in records])
        frames=np.flatnonzero(mask)
        roots=np.concatenate([data['root_position'],data['root_rotation_xyzw'],
                              data['body_linear_velocity'][:,:,0],data['body_angular_velocity'][:,:,0]],axis=-1)
        boxes=np.concatenate([data['box_position'],data['box_rotation_xyzw'],
                              data['box_linear_velocity'],data['box_angular_velocity']],axis=-1)
        np.savez_compressed(path/'screened_rsi_snapshots.npz',source_frame=frames,source_time=data['time'][frames],
                            root_state=roots[frames],dof_position=data['dof_position'][frames],
                            dof_velocity=data['dof_velocity'][frames],box_state=boxes[frames,None,:],
                            box_size=data['box_size'],short_rsi_screen_pass=np.ones(len(frames),dtype=bool),
                            screen_seconds=.1,training_loader_integrated=False,continuous_motion_validated=False)
        selected[skill]=len(frames)
        if not args.no_render:
            before=dict(np.load(args.input/skill/'cooperative_rsi_candidate.npz'))
            geometry=np.load(args.before_audit/skill/'cooperative/simulator_reset_geometry.npz')
            before['body_position']=geometry['body_position']
            before['body_rotation_xyzw']=geometry['body_rotation_xyzw']
            before['render_title']='BEFORE / '+SKILL_TITLES[skill]
            before['render_caption']='Actual 32-DOF pose | original inward placement'
            before['render_footer']='Uncorrected | hand / forearm penetration'
            data['render_title']='CORRECTED / '+SKILL_TITLES[skill]
            data['render_footer']='Screened RSI subset | continuous gait not validated'
            dest=args.output/'before_after'/skill;dest.mkdir(parents=True,exist_ok=True)
            render(before,data,dest)
    if not args.no_render:
        dest=args.output/'before_after'
        for video in ('comparison','comparison_half_speed'):
            playlist=dest/(video+'_playlist.txt')
            playlist.write_text(''.join("file '{}/{}.mp4'\n".format(skill,video) for skill in SKILL_TAGS))
            subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(playlist),
                            '-c','copy','-movflags','+faststart',str(dest/(video+'_all_skills.mp4'))],check=True)
    (args.output/'selected_frames.json').write_text(json.dumps(selected,indent=2)+'\n')
    readme='''Corrected cooperative RSI snapshots

Each skill has a complete corrected reference and screened_rsi_snapshots.npz.
Snapshots are independent initial states, not a concatenated expert trajectory.
The correction uses physical 32-DOF elbows/wrists, arm IK, and a constant 1.4-2.6cm
root lift per clip. Leg/torso/head joint rotations are unchanged. Box dimensions
and the 15cm interior grip placement are retained. Arm DOFs can change up to
42 degrees: this is not an unchanged-joint rigid translation.

Snapshot root_state: [sample,2,13] = xyz, quaternion xyzw, linear velocity xyz,
angular velocity xyz. dof_position/dof_velocity: [sample,2,32].
box_state: [sample,1,13], one shared 16.64kg box at density 100kg/m^3.
source_frame/source_time identify the frame in the full corrected reference.

Screen: native asset PD at the initial pose, free roots, CPU PhysX 0.1s,
repository shock thresholds plus <=2cm initial geometry penetration and original
RSI exclusions. CarryWith also requires both people to make hand contact for
at least 3/6 steps. The report describes geometry approximations and limits.

This is a short reset screen, not long-term stability certification. Clone foot
sliding remains in the continuous reference. Native-PD playback still loses
balance; no trained balancing controller was used. Current training reset does
not support this paired snapshot format yet. No loader or training was changed.
Use the snapshot state arrays, not the full video, as candidate initial states.
'''
    (args.output/'README.txt').write_text(readme)
    with zipfile.ZipFile(args.output/'corrected_rsi_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('README.txt','selected_frames.json','correction_summary.json'):
            archive.write(args.output/name,name)
        for skill in SKILL_TAGS:
            for name in ('cooperative_rsi_candidate.npz','screened_rsi_snapshots.npz','metadata.json'):
                archive.write(args.output/skill/name,str(Path(skill)/name))
            for name in ('summary.json','frames.csv','screen_masks.npz'):
                archive.write(args.audit/skill/'cooperative'/name,str(Path('physics')/skill/name))
        archive.write(args.audit/'report.json','physics/report.json')
    print('Packaged screened snapshots:',selected,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('joint_carry/archive/carry_joint_preview_all_inset15'))
    parser.add_argument('--output',type=Path,default=Path('joint_carry/carry_joint_corrected'))
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--finalize',action='store_true',help='Package existing candidates after a matching physics audit.')
    parser.add_argument('--audit',type=Path,default=Path('joint_carry/carry_corrected_physics'))
    parser.add_argument('--before-audit',type=Path,default=Path('joint_carry/archive/carry_physics_validation'))
    args=parser.parse_args()
    if args.finalize:
        finalize(args)
        return
    summary={}
    for skill in SKILL_TAGS:
        source=args.input/skill
        output=args.output/skill;output.mkdir(parents=True,exist_ok=True)
        data=dict(np.load(source/'cooperative_rsi_candidate.npz',allow_pickle=False))
        corrected,stats=correct(data)
        np.savez_compressed(output/'cooperative_rsi_candidate.npz',**corrected)
        shutil.copyfile(source/'original_joints.npz',output/'original_joints.npz')
        metadata=dict(skill=skill,input=str(source),ik_applied=True,simulator_pose_exact=True,
                      rsi_ready=False,physics_validated=False,stats=stats,
                      limitations=['Clone gait sliding remains in the continuous reference.',
                                   'Use separately screened RSI frames; not a continuous expert motion.',
                                   'Three independent clips; no trained controller or reset integration.'])
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        summary[skill]={k:v for k,v in stats.items() if k!='arm_fit'}
        print(skill,summary[skill],flush=True)
        if not args.no_render:
            render(dict(np.load(output/'original_joints.npz')),corrected,output)
    (args.output/'correction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not args.no_render:
        for video in ('comparison','cooperative','comparison_half_speed'):
            playlist=args.output/(video+'_playlist.txt')
            playlist.write_text(''.join("file '{}/{}.mp4'\n".format(skill,video) for skill in SKILL_TAGS))
            subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(playlist),
                            '-c','copy','-movflags','+faststart',str(args.output/(video+'_all_skills.mp4'))],check=True)


if __name__=='__main__':
    main()
