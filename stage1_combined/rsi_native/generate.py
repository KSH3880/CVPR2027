from pathlib import Path
import sys,argparse,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from utils.motion_lib import MotionLib
from rsi_native.geometry import NativeGeometry,collision_samples,rotate


def assets():
    c=json.loads((ROOT/'chair_assets/catalog.json').read_text());result=[]
    for e in c['train']:
        result.append(dict(id=e['id'],category='Chair',split='train',path=e['path'],bbox=e['bbox'],seat=e['seat'],extent=e['seat_extent'],feature=e['feature']))
    for e in c['tables']:
        p=Path(e['path']);d=json.loads((p/'config.json').read_text());seat=[0,0,e['bbox'][2]/2]
        result.append(dict(id=p.name,category='Table',split='train',path=str(p),bbox=e['bbox'],seat=seat,extent=e['bbox'][:2],feature=[1,0,0,*seat]))
    for e in result:
        d=json.loads((Path(e['path'])/'config.json').read_text());e['metadata_height']=d['tarSitPos'][2];e['facing']=d['facing'][:2]
    return result


def references(skill,samples):
    lib=MotionLib('tokenhsi/data/dataset_loco_sit_carry_climb.yaml',skill,[1,2,3,4,6,7,9,10,11,12,13,14],[0,3,6,9,10,13,14,17,20,23,26,29,32],[5,8,11,14],'cpu')
    names=lib._motions[0].skeleton_tree.node_names;result=[]
    for clip in range(lib.num_motions()):
        count=int(lib._motion_num_frames[clip]);ids=torch.full((count,),clip,dtype=torch.long);times=torch.arange(count)*lib._motion_dt[clip]
        root,q,dof,vel,ang,dv,feet=lib.get_motion_state(ids,times);body,bq,bv,ba=lib.get_motion_state_max(ids,times)
        box,_=lib.get_obj_motion_state_single_frame(ids[:1])
        near=(root[:,:2]-box[0,:2]).norm(dim=-1)<.8
        speed=vel.norm(dim=-1)
        if skill=='sit':
            eligible=near&(speed<.5)&(root[:,2]<.86)
            if eligible.any():
                low=torch.quantile(root[eligible,2],.2)
                phase=eligible&(root[:,2]<=low+.05)
                surface=float(root[phase,2].median())-.1
            else:
                phase=torch.zeros(count,dtype=torch.bool);surface=float(root[:,2].min())-.1
        else:
            footheight=feet[:,-2:,2].mean(-1)
            footspeed=bv[:,[11,14]].norm(dim=-1).amax(-1)
            eligible=near&(speed<.8)&(footspeed<.6)&(footheight>.15)&((feet[:,-2:,2].amax(-1)-feet[:,-2:,2].amin(-1))<.10)
            if eligible.any():
                top=torch.quantile(footheight[eligible],.8)
                phase=eligible&(footheight>=top-.05)
                surface=float(footheight[phase].median())
            else:
                phase=torch.zeros(count,dtype=torch.bool);surface=float(footheight.max())
        cost=(root[:,2]-(surface+(.1 if skill=='sit' else .94))).square()+(root[:,:2]-box[0,:2]).square().sum(-1)
        anchor=int(cost.argmin());anchor_valid=bool(phase.any())
        if anchor_valid:anchor=int(torch.where(phase,cost,torch.full_like(cost,float('inf'))).argmin())
        selected=torch.unique(torch.cat([torch.linspace(0,anchor,max(2,samples//2)).long(),torch.linspace(max(0,anchor-30),min(count-1,anchor+30),max(2,samples//2)).long()])).numpy()
        heading=rotate(q[anchor].numpy(),np.array([1.,0.,0.]));yaw=math.atan2(heading[1],heading[0])
        result.append(dict(clip=clip,source=str(lib._motion_files[clip]),anchor=anchor,anchor_valid=anchor_valid,anchor_xy=root[anchor,:2].numpy(),yaw=yaw,ref_surface=surface,frame=selected,time=times[selected].numpy(),phase=phase[selected].numpy(),root=root[selected].numpy(),quat=q[selected].numpy(),dof=dof[selected].numpy(),vel=vel[selected].numpy(),ang=ang[selected].numpy(),dof_vel=dv[selected].numpy(),body=body[selected].numpy(),body_rot=bq[selected].numpy(),body_vel=bv[selected].numpy(),body_ang=ba[selected].numpy()))
    return result,list(names)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples-per-clip',type=int,default=24);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);entries=assets();refs={};names=None
    for skill in ['sit','climb']:refs[skill],names=references(skill,a.samples_per_clip)
    shapes=collision_samples(ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',names)
    report=[]
    for index,e in enumerate(entries):
        geo=NativeGeometry(e['path'],shapes);folder=out/e['id'];folder.mkdir()
        surface_z=e['bbox'][2]/2+e['seat'][2]
        for skill in ['sit','climb']:
            data={k:[] for k in ['root_state','dof','dof_vel','body_state','clip','frame','time','phase','anchor_valid','transform']}
            for ref in refs[skill]:
                facing=e['facing'] if e['category']=='Chair' else [1,0]
                yaw=math.atan2(facing[1],facing[0])-ref['yaw'];rot=Rotation.from_euler('z',yaw);q=rot.as_quat().astype(np.float32)
                anchor=np.r_[ref['anchor_xy'],0.];translation=np.r_[e['seat'][:2],surface_z-ref['ref_surface']]-rot.apply(anchor)
                pos=rot.apply(ref['root'])+translation;quat=(rot*Rotation.from_quat(ref['quat'])).as_quat();vel=rot.apply(ref['vel']);ang=rot.apply(ref['ang'])
                bp=rot.apply(ref['body'].reshape(-1,3)).reshape(ref['body'].shape)+translation
                bq=(rot*Rotation.from_quat(ref['body_rot'].reshape(-1,4))).as_quat().reshape(ref['body_rot'].shape)
                bv=rot.apply(ref['body_vel'].reshape(-1,3)).reshape(ref['body_vel'].shape);ba=rot.apply(ref['body_ang'].reshape(-1,3)).reshape(ref['body_ang'].shape)
                n=len(pos);data['root_state'].append(np.c_[pos,quat,vel,ang]);data['dof'].append(ref['dof']);data['dof_vel'].append(ref['dof_vel']);data['body_state'].append(np.concatenate([bp,bq,bv,ba],-1))
                for k in ['frame','time','phase']:data[k].append(ref[k])
                data['clip'].append(np.full(n,ref['clip']));data['anchor_valid'].append(np.full(n,ref['anchor_valid']));data['transform'].append(np.tile(np.r_[q,translation],(n,1)))
            data={k:np.concatenate(v) for k,v in data.items()};clearance,floor=geo.check(data['body_state'][...,:3],data['body_state'][...,3:7],e['bbox'][2]/2)
            finite=np.isfinite(data['root_state']).all(-1)&np.isfinite(data['dof']).all(-1)
            static=finite&data['anchor_valid']&(clearance>=-.015)&(floor>=-.02)
            correction=np.zeros((len(clearance),3),dtype=np.float32)
            facing=np.asarray(e['facing'] if e['category']=='Chair' else [1.,0.],dtype=np.float32);facing/=np.linalg.norm(facing)
            # Bounded rigid adjustments only; reject if no collision-safe near-goal pose exists.
            offsets=sorted([(front,lift) for front in [0.,.075,.15] for lift in [0.,.025,.05,.075,.1]],key=lambda x:x[0]**2+x[1]**2)
            goal=np.r_[e['seat'][:2],surface_z+(.1 if skill=='sit' else .94)]
            for front,lift in offsets:
                chosen=np.flatnonzero(data['phase']&data['anchor_valid']&~static)
                if not len(chosen):break
                delta=np.r_[facing*front,lift]
                shifted=data['body_state'][chosen,:,:3]+delta
                depth,ground=geo.check(shifted,data['body_state'][chosen,:,3:7],e['bbox'][2]/2)
                goalerror=np.linalg.norm(data['root_state'][chosen,:3]+delta-goal,axis=-1)
                ok=(depth>=-.015)&(ground>=-.02)&(goalerror<=.20)
                good=chosen[ok]
                data['root_state'][good,:3]+=delta;data['body_state'][good,:,:3]+=delta
                data['transform'][good,4:]+=delta;correction[good]=delta
                clearance[good]=depth[ok];floor[good]=ground[ok];static[good]=True
            data.update(clearance=clearance,floor_clearance=floor,static_pass=static,correction=correction)
            np.savez_compressed(folder/(skill+'.npz'),**data)
            row=dict(asset=e['id'],category=e['category'],skill=skill,candidates=len(clearance),static_pass=int(data['static_pass'].sum()),phase_pass=int((data['static_pass']&data['phase']).sum()),floor_rejected=int((floor<-.02).sum()),overlap_rejected=int((clearance<-.015).sum()),mesh_watertight=geo.watertight,min_clearance=float(clearance.min()),min_floor=float(floor.min()))
            report.append(row);print('RSI_GEOMETRY',index+1,len(entries),json.dumps(row),flush=True)
        (out/'geometry_report.json').write_text(json.dumps(report,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(assets=entries,body_names=names,samples_per_clip=a.samples_per_clip,recipe='constant yaw/XY/support-height rigid retarget; reference support inferred from stable seated pelvis-.10 or standing feet; object_state z is ground origin; no joint IK; phase-only rigid forward0/.075/.15m and lift0..10cm search with goal error<=20cm',reference_counts={k:len(v) for k,v in refs.items()},geometry='sphere/capsule centerline<=15mm sampling + foot vertices/face-centers, Open3D triangle distance/parity; open meshes not exact-volume-certified',thresholds=dict(penetration_proxy=.015,floor=.02),source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('geometry.py')]}),indent=2));print('GENERATED',out,flush=True)
if __name__=='__main__':main()
