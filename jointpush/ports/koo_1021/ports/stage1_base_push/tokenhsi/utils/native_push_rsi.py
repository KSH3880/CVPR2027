"""Unmodified PUSH phase selection and conservative collider-safe reset placement."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from utils.door_reference import rotate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_contract(env):
    import yaml
    manifest=Path(env['motion_file'])
    asset=Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
    # Mesh contents are part of the geometry contract, too.
    meshes=[asset.parent/m.get('file') for m in ET.parse(asset).findall('./asset/mesh')]
    rows=yaml.safe_load(manifest.read_text())['motions']['push']
    return dict(revision=1,asset=digest(asset),meshes=[digest(p) for p in meshes],
        motions=[digest(manifest.parent/r['file']) for r in rows],box=env['pushBox'],
        scene=env['pushScene'],reset=env['box']['reset'],reward=env['relationReward'],
        rsi=env['openPushRSI'],selection=dict(height=1.,reach=.25,drop=.25,trim=.2,min_duration=.5),
        placement='rigid_yaw_xy_minimum_floor_z_collider_retreat_v1')


def geometry(asset,names,device):
    """Sphere/capsule covers for box safety; exact foot vertices for floor height."""
    from utils.manipulation_contact import collision_samples
    samples=collision_samples(asset,include_hands=True)
    tree=ET.parse(asset);asset=Path(asset)
    meshes={m.get('name'):asset.parent/m.get('file') for m in tree.findall('./asset/mesh')}
    ground=[]
    for body in tree.findall('.//body'):
        for g in body.findall('./geom'):
            if g.get('type')=='mesh':
                offset=torch.tensor([float(v) for v in g.get('pos','0 0 0').split()])
                for line in meshes[g.get('mesh')].read_text().splitlines():
                    if line.startswith('v '):ground.append((names.index(body.get('name')),(torch.tensor([float(v) for v in line.split()[1:4]])+offset).tolist(),0.))
            elif 'fromto' in g.attrib:
                ends=[float(v) for v in g.get('fromto').split()]
                for p in (ends[:3],ends[3:]):ground.append((names.index(body.get('name')),p,float(g.get('size').split()[0])))
            else:
                ground.append((names.index(body.get('name')),[float(v) for v in g.get('pos','0 0 0').split()],float(g.get('size').split()[0])))
    return dict(ids=torch.tensor([names.index(n) for n,_,_ in samples],device=device),
        offsets=torch.tensor([p for _,p,_ in samples],device=device),radii=torch.tensor([r for _,_,r in samples],device=device),
        ground_ids=torch.tensor([n for n,_,_ in ground],device=device),ground_offsets=torch.tensor([p for _,p,_ in ground],device=device),
        ground_radii=torch.tensor([r for _,_,r in ground],device=device),
        hands=[names.index('right_hand'),names.index('left_hand')],
        foot_masks=[torch.tensor([n==names.index(f) for n,_,_ in ground],device=device) for f in ('right_foot','left_foot')])


def safe_place(body,rot,box,size,targets,geo,return_metrics=False):
    """Only rigid body translation; preserve every original DOF and velocity."""
    body=body.clone()
    floor=body[:,geo['ground_ids']]+rotate(rot[:,geo['ground_ids']],geo['ground_offsets'].expand(len(body),-1,-1))
    dz=(-((floor[...,2]-geo['ground_radii']).amin(-1))).clamp_min(0)
    body[...,2]+=dz[:,None]
    points=body[:,geo['ids']]+rotate(rot[:,geo['ids']],geo['offsets'].expand(len(body),-1,-1))
    tangent=targets[:,0,:2]-targets[:,1,:2]
    tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
    center=targets[:,:,:2].mean(1)
    # Determine the actual width of the selected face in the box local frame.
    from utils.door_reference import yaw_from_quat
    yaw=yaw_from_quat(box[:,3:7]);c,s=torch.cos(yaw),torch.sin(yaw)
    local_t=torch.stack((c*tangent[:,0]+s*tangent[:,1],-s*tangent[:,0]+c*tangent[:,1]),-1)
    halfwidth=(local_t.abs()*size[:,:2]).sum(-1)/2
    along=((points[...,:2]-center[:,None])*tangent[:,None]).sum(-1)
    front=((points[...,:2]-center[:,None])*normal[:,None]).sum(-1)+geo['radii']
    overlap=(along.abs()<=halfwidth[:,None]+geo['radii']) & (points[...,2]+geo['radii']>=0) & (points[...,2]-geo['radii']<=size[:,2,None])
    retreat=(front+.005).masked_fill(~overlap,0).amax(-1).clamp_min(0)
    body[...,:2]-=retreat[:,None,None]*normal[:,None]
    hands=body[:,geo['hands']]
    gap=-((hands[...,:2]-center[:,None])*normal[:,None]).sum(-1)
    hand_along=((hands[...,:2]-center[:,None])*tangent[:,None]).sum(-1).abs()
    metrics=dict(z_shift=dz,retreat=retreat,hand_gap=gap,
        placed_hand_midpoint_z=hands[...,2].mean(-1),
        face_valid=(hand_along+.05<=halfwidth[:,None]).all(-1),
        sole_height_m=torch.stack([floor[:,mask,2].amin(-1)+dz for mask in geo['foot_masks']],-1))
    return (body,metrics) if return_metrics else body


def phase_mask(body,rot,names,fps,history=.3):
    hands=[names.index('right_hand'),names.index('left_hand')]
    shoulders=[names.index('right_upper_arm'),names.index('left_upper_arm')]
    # Asset torso +X, projected onto the horizontal plane.
    torso=names.index('torso')
    forward=rotate(rot[:,torso],body.new_tensor([1.,0.,0.]).expand(len(body),3))[:,:2]
    forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    delta=body[:,hands]-body[:,shoulders]
    valid=(body[:,hands,2].mean(-1)<=1.) & ((delta[...,:2]*forward[:,None]).sum(-1)>=.25).all(-1) & (delta[...,2]>=-.25).all(-1)
    valid &= torch.arange(len(body),device=body.device)>=math.ceil(history*fps)
    return valid


def trim_runs(valid,fps):
    result=torch.zeros_like(valid);trim=math.ceil(.2*fps)
    indices=valid.nonzero().flatten().cpu().tolist()
    runs=[]
    for f in indices:
        if not runs or f!=runs[-1][-1]+1:runs.append([])
        runs[-1].append(f)
    for run in runs:
        kept=run[trim:len(run)-trim]
        if len(kept)>=2 and (kept[-1]-kept[0])/fps>=.5-1e-7:result[kept]=True
    return result


class NativePushBank:
    def __init__(self,spec,libraries,env):
        payload=json.loads(Path(spec['bank_file']).read_text())
        if payload['contract']!=native_contract(env):raise ValueError('Native PUSH RSI contract mismatch')
        if payload.get('screen')!='PD_0.1s_all_profiles' and not env.get('_native_geometry_check',False):
            raise ValueError('Native PUSH RSI needs an explicit CPU physics screen before training')
        self.spec=spec;self.rows={'push':payload['candidates']['push']};lib=libraries['push']
        names=lib._motions[0].skeleton_tree.node_names
        self.geometry=geometry(Path(env['asset']['assetRoot'])/env['asset']['assetFileName'],names,lib.gts.device)
        self.device=lib.gts.device
        rows=self.rows['push']
        weights=torch.zeros(len(rows),device=self.device);late=torch.zeros_like(weights)
        for mid in sorted(set(r['motion_id'] for r in rows)):
            indices=[i for i,r in enumerate(rows) if r['motion_id']==mid]
            pool=[rows[i] for i in indices]
            if any(not all(math.isfinite(r[k]) for k in ('time','phase','angle','box_progress')) or r['time']<.3 or r['time']>float(lib._motion_lengths[mid])+1e-5 for r in pool):raise ValueError('Invalid native RSI candidate')
            weight=float(lib._motion_weights[mid])
            weights[indices]=weight/len(indices)
            tail=indices[int(len(indices)*(1-spec['late_fraction'])):]
            late[tail]=weight/len(tail)
        if not len(rows) or not weights.sum():raise ValueError('Empty PUSH phase pool; no fallback allowed')
        if any(r['motion_id'] in spec.get('excluded_motion_ids',[]) for r in rows):
            raise ValueError('Excluded clip in native PUSH bank')
        self.weights,self.late=weights,late
        self.values={k:torch.tensor([r[k] for r in rows],device=self.device) for k in ('motion_id','time','phase','angle','box_progress')}

    def sample(self,skill,count):
        if skill!='push':raise ValueError('Native phase bank is PUSH-only')
        chosen=torch.multinomial(self.weights,count,replacement=True)
        requested=torch.rand(count,device=self.device)<self.spec['late_probability']
        replacement=torch.multinomial(self.late,count,replacement=True)
        chosen=torch.where(requested,replacement,chosen)
        result={k:v[chosen] for k,v in self.values.items()}
        result['late_requested']=requested
        return result


def load_cpu_reference(env):
    """Load the actual local MotionLib and geometry without creating a simulator."""
    from utils.motion_lib import MotionLib
    asset=Path(env['asset']['assetRoot'])/env['asset']['assetFileName']
    bodies=ET.parse(asset).findall('.//worldbody//body');names=[b.get('name') for b in bodies]
    dofs=[];offsets=[0]
    for i,b in enumerate(bodies):
        joints=b.findall('./joint')
        if joints:dofs.append(i);offsets.append(offsets[-1]+len(joints))
    hands=[names.index('right_hand'),names.index('left_hand')]
    lib=MotionLib(env['motion_file'],'push',dofs,offsets,hands+[names.index('right_foot'),names.index('left_foot')],'cpu')
    return lib,names,geometry(asset,names,'cpu')
