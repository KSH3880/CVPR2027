"""Chair-front retarget/filter and exact foot-surface crossing rejection."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from rsi_native.geometry import collision_samples,NativeGeometry
from rsi_native.support import supported
from rsi_native.foot_surface import FootSurface


def front_mask(root_xy,seat_xy,facing,central=.12):
    f=np.asarray(facing,dtype=float);f/=np.linalg.norm(f);delta=root_xy-np.asarray(seat_xy)
    forward=delta@f;lateral=delta@np.array([-f[1],f[0]])
    return (np.linalg.norm(delta,axis=-1)<=central)|((forward>=.03)&(np.abs(lateral)<=forward))


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 m=json.loads((a.source/'manifest.json').read_text());xml='tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml';shapes=collision_samples(xml,m['body_names']);feet=FootSurface(xml,m['body_names']);report=[]
 for e in m['assets']:
  geo=NativeGeometry(e['path'],shapes);folder=a.output/e['id'];folder.mkdir()
  for skill in ['sit','climb']:
   d=dict(np.load(a.source/e['id']/(skill+'.npz')));front=np.ones(len(d['clip']),dtype=bool);turned=np.zeros_like(front)
   if e['category']=='Chair':
    f=np.asarray(e['facing'],dtype=float);f/=np.linalg.norm(f);delta=d['root_state'][:,:2]-np.asarray(e['seat'][:2]);turned=(delta@f<-.03)&(np.linalg.norm(delta,axis=-1)>.12)
    if turned.any():
     rot=Rotation.from_euler('z',np.pi);center=np.r_[e['seat'][:2],0.];r=d['root_state'][turned].copy();b=d['body_state'][turned].copy()
     r[:,:3]=rot.apply(r[:,:3]-center)+center;r[:,3:7]=(rot*Rotation.from_quat(r[:,3:7])).as_quat()
     for k in [7,10]:r[:,k:k+3]=rot.apply(r[:,k:k+3]);b[:,:,k:k+3]=rot.apply(b[:,:,k:k+3].reshape(-1,3)).reshape(-1,15,3)
     b[:,:,:3]=rot.apply((b[:,:,:3]-center).reshape(-1,3)).reshape(-1,15,3)+center;b[:,:,3:7]=(rot*Rotation.from_quat(b[:,:,3:7].reshape(-1,4))).as_quat().reshape(-1,15,4)
     d['root_state'][turned]=r;d['body_state'][turned]=b
     transform=d['transform'][turned].copy();transform[:,:4]=(rot*Rotation.from_quat(transform[:,:4])).as_quat();transform[:,4:]=rot.apply(transform[:,4:]-center)+center;d['transform'][turned]=transform
    front=front_mask(d['root_state'][:,:2],e['seat'][:2],e['facing'])
   cross=feet.crossing(d['body_state'],geo,e['bbox'][2]/2);clearance,floor=geo.check(d['body_state'][...,:3],d['body_state'][...,3:7],e['bbox'][2]/2);support,gap=supported(d['body_state'],geo,e['bbox'][2]/2)
   valid=d['anchor_valid']&(d['dof_correction_max']<=.3)&(clearance>=-.015)&(floor>=-.02)&support&front&~cross
   d.update(static_pass=valid,clearance=clearance,floor_clearance=floor,support_pass=support,support_gap=gap,front_pass=front,foot_crossing=cross,front_turned=turned)
   np.savez_compressed(folder/(skill+'.npz'),**d);row=dict(asset=e['id'],category=e['category'],skill=skill,static=int(valid.sum()),phase=int((valid&d['phase']).sum()),nonphase=int((valid&~d['phase']).sum()),foot_cross_rejected=int(cross.sum()),front_rejected=int((~front).sum()),turned=int(turned.sum()));report.append(row);print('FRONT_FOOT',json.dumps(row),flush=True)
 m.update(chair_front_only=True,front_recipe='approach ±45deg front cone; central seated/on-seat disk12cm exempt; rear approach rigidly turned180deg about seat then rechecked',foot_triangle_tolerance=.002);(a.output/'manifest.json').write_text(json.dumps(m,indent=2));(a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
