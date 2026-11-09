import math
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from push_reference.utils.jointpush_geometry import rotate, multiply, yaw_from_quat


def collision_samples(asset_file, include_hands=False):
    """Cover spheres/capsules and foot mesh AABBs with local bounding spheres."""
    asset_file = Path(asset_file)
    tree = ET.parse(asset_file)
    meshes = {m.attrib['name']: asset_file.parent / m.attrib['file']
              for m in tree.findall('./asset/mesh')}
    result = []
    for body in tree.findall('.//body'):
        name = body.attrib['name']
        if not include_hands and name in ('right_hand', 'left_hand'):
            continue
        for geom in body.findall('./geom'):
            kind = geom.get('type', 'capsule')
            if kind == 'mesh':
                vertices = [list(map(float, line.split()[1:4]))
                            for line in meshes[geom.attrib['mesh']].read_text().splitlines()
                            if line.startswith('v ')]
                points = torch.tensor(vertices)
                lo, hi = points.amin(0), points.amax(0)
                center = (lo + hi) / 2
                center += torch.tensor(list(map(float, geom.get('pos', '0 0 0').split())))
                result.append((name, center.tolist(), float((hi - lo).norm() / 2)))
            elif 'fromto' in geom.attrib:
                ends = torch.tensor(list(map(float, geom.attrib['fromto'].split()))).reshape(2, 3)
                radius = float(geom.attrib['size'].split()[0])
                length = float((ends[1] - ends[0]).norm())
                count = max(2, math.ceil(length / radius) + 1)
                cover = math.sqrt(radius ** 2 + (length / (count - 1) / 2) ** 2)
                for t in torch.linspace(0, 1, count):
                    result.append((name, ((1 - t) * ends[0] + t * ends[1]).tolist(), cover))
            elif kind == 'sphere':
                result.append((name, list(map(float, geom.get('pos', '0 0 0').split())),
                               float(geom.attrib['size'])))
            else:
                raise ValueError('Unsupported contact geometry: ' + kind)
    if not result:
        raise ValueError('No non-hand contact geometry')
    return result



def geometry(asset,names,device):
    """Sphere/capsule covers for box safety; exact foot vertices for floor height."""
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



def align_push_pose(body, body_rot, body_vel, body_ang, targets, hands, stand_off):
    """Fit both hand XY points with one rigid yaw/translation; retain foot Z."""
    delta=body[:,hands[0],:2]-body[:,hands[1],:2]
    wanted_delta=targets[:,0,:2]-targets[:,1,:2]
    yaw=torch.atan2(wanted_delta[:,1],wanted_delta[:,0])-torch.atan2(delta[:,1],delta[:,0])
    q=body_rot[:,0].new_zeros(len(body),4);q[:,2]=torch.sin(yaw/2);q[:,3]=torch.cos(yaw/2)
    expanded=q[:,None].expand_as(body_rot)
    relative=rotate(expanded,body-body[:,:1])
    right=wanted_delta/wanted_delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    normal=torch.stack((-right[:,1],right[:,0]),-1)
    root=body[:,0].clone()
    root[:,:2]=targets[:,:,:2].mean(1)-relative[:,hands,:2].mean(1)-stand_off*normal
    return (relative+root[:,None],multiply(expanded,body_rot),
            rotate(expanded,body_vel),rotate(expanded,body_ang),q)
