"""Episode-fixed, box-local hand targets for one-face or adjacent-face Push."""
import math
import torch


def rotate(q, v):
    """xyzw unit quaternion rotation, supporting arbitrary point axes."""
    while q.ndim < v.ndim:
        q=q.unsqueeze(1)
    xyz=q[...,:3].expand_as(v)
    t=2*torch.cross(xyz,v,dim=-1)
    return v+q[...,3:4]*t+torch.cross(xyz,t,dim=-1)


def world_targets(boxes, local_points):
    center=boxes[:,:3]
    while center.ndim < local_points.ndim:
        center=center.unsqueeze(1)
    return rotate(boxes[:,3:7],local_points)+center


def assign_contact_targets(boxes, goals, sizes, roots, spec, height=.8):
    """Choose reset-time faces, then minimize total agent-to-approach distance.

    Input hand order is right, left. Local targets persist until that env resets.
    Angle is relative to the nearest horizontal box axis (0..45 degrees).
    """
    expected={'split_angle_deg','hand_spacing','same_face_offset_fraction','approach_standoff'}
    if set(spec)!=expected or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in spec.values()):
        raise ValueError('Invalid assigned contact specification')
    if not 0<spec['split_angle_deg']<45 or not 0<spec['same_face_offset_fraction']<.5 or min(spec['hand_spacing'],spec['approach_standoff'])<=0:
        raise ValueError('Invalid contact assignment bounds')
    if roots.shape[1]!=2:
        raise ValueError('Assigned faces require exactly two agents')
    if (sizes[:,2]<=height).any():raise ValueError('Hand target must be below box top')
    inverse=boxes[:,3:7].clone();inverse[:,:3]*=-1
    direction=rotate(inverse,goals-boxes[:,:3])[:,:2]
    angle=torch.atan2(direction.abs().amin(-1),direction.abs().amax(-1).clamp_min(1e-6))
    two=angle>math.radians(spec['split_angle_deg'])
    axis=(direction[:,1].abs()>=direction[:,0].abs()).long()
    second=torch.where(two,1-axis,axis)
    axes=torch.stack((axis,second),-1)
    signs=torch.where(direction.gather(1,axes)>=0,1.,-1.)
    normals=torch.zeros(boxes.shape[0],2,2,device=boxes.device,dtype=boxes.dtype).scatter_(2,axes[...,None],signs[...,None])
    tangent=torch.stack((-normals[...,1],normals[...,0]),-1)
    half=sizes[:,:2]/2
    center=-normals*half[:,None]
    width=sizes[:,:2].gather(1,1-axes)
    offsets=boxes.new_tensor([-1.,1.])[None]*width*spec['same_face_offset_fraction']
    center=center+torch.where(two[:,None],torch.zeros_like(offsets),offsets)[...,None]*tangent
    # Never place a hand beyond a finite face, including narrow rectangle sides.
    margin=width/2-torch.where(two[:,None],torch.zeros_like(offsets),offsets.abs())
    if (margin<spec['hand_spacing']/2).any():raise ValueError('Hand span exceeds assigned face interior')
    local_hands=boxes.new_zeros((boxes.shape[0],2,2,3))
    local_hands[...,:2]=center[:,:,None]+boxes.new_tensor([-1.,1.])[None,None,:,None]*spec['hand_spacing']/2*tangent[:,:,None]
    local_hands[...,2]=height-sizes[:,None,None,2]/2
    approach=boxes.new_zeros((boxes.shape[0],2,3))
    approach[...,:2]=center-normals*spec['approach_standoff']
    world=world_targets(boxes,approach)
    costs=torch.cdist(roots[...,:2],world[...,:2])
    swap=(costs[:,0,1]+costs[:,1,0])<(costs[:,0,0]+costs[:,1,1])
    order=torch.where(swap[:,None],boxes.new_tensor([1,0],dtype=torch.long),boxes.new_tensor([0,1],dtype=torch.long))
    rows=torch.arange(boxes.shape[0],device=boxes.device)[:,None]
    return dict(local_hands=local_hands[rows,order],local_approach=approach[rows,order],
                local_normals=normals[rows,order],two_faces=two,angle_deg=angle*180/math.pi)
