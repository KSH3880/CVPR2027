"""Material anchor observations and face contact without midpoint objectives."""
import math
import torch
from utils.door_reference import rotate


def validate_point(env):
    spec=env['pushContactPoint']
    if set(spec)!={'height_range_m','rsi_clip_coverage'} or spec['rsi_clip_coverage']!='pd_survivors':
        raise ValueError('Invalid PUSH point fields')
    bounds=spec['height_range_m']
    if not isinstance(bounds,list) or len(bounds)!=2:
        raise ValueError('Point height requires two bounds')
    values=bounds
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Point settings must be finite and positive')
    if not bounds[0]<=bounds[1]<env['pushBox']['size'][2]:
        raise ValueError('Invalid point height/tolerance range')
    if not env.get('pushOnly') or env['relationReward']['push'].get('contact_mode')!='fixed_face_surface':
        raise ValueError('Point task requires fixed-face PUSH-only profile')
    if 'pushSizeRandomization' in env:
        raise ValueError('This point experiment creates fixed-size boxes')
    if 'tilt_fade_start' in env['relationReward']['contact_at']:
        raise ValueError('Point experiment preserves the no-fade AT objective')
    return spec


def local_points(side,size,height):
    """A face material point, independent of hands and current goal direction."""
    size=torch.as_tensor(size,device=side.device,dtype=height.dtype).expand(*side.shape,3)
    point=torch.zeros_like(size)
    axis=(side>=2).long()
    sign=torch.where(side.remainder(2)==0,-1.,1.)
    point.scatter_(-1,axis[...,None],(sign*size.gather(-1,axis[...,None]).squeeze(-1)/2)[...,None])
    point[...,2]=height-size[...,2]/2
    return point


def world_points(boxes,local):
    return boxes[...,:3]+rotate(boxes[...,3:7],local)


def anchor_view(obs,num_agents,entity_sizes,point_local,pose_size=7):
    """Replace object center pose/velocity with its material anchor, like graspview.

    Physical bbox, angular velocity, human/AT poses and graph packet stay intact.
    This PUSH graph has one separately bound object per owner, so a single
    scene already contains both owner-specific anchors; no duplicated views.
    """
    h,o,g=entity_sizes;m=num_agents
    result=obs.clone()
    node_end=m*h+m*o+m*g
    objects=result[:,m*h:m*h+m*o].reshape(len(obs),m,o)
    poses=result[:,node_end:node_end+3*m*pose_size].reshape(len(obs),3*m,pose_size)
    objects[...,:3]+=torch.cross(objects[...,3:6],point_local,dim=-1)
    poses[:,m:2*m,:3]+=rotate(poses[:,m:2*m,3:7],point_local)
    return result


def point_geometry(hands,boxes,size,side,local):
    size=torch.as_tensor(size,device=boxes.device,dtype=boxes.dtype).expand(*boxes.shape[:2],3)
    inv=boxes[...,3:7].clone();inv[...,:3]*=-1
    hand_local=rotate(inv[:,:,None].expand(-1,-1,2,-1),hands-boxes[:,:,None,:3])
    axis=(side>=2).long()
    other=1-axis
    across=(hand_local.gather(-1,axis[...,None,None].expand(-1,-1,2,1)).squeeze(-1)
            -local.gather(-1,axis[...,None]).squeeze(-1)[...,None])
    along=hand_local.gather(-1,other[...,None,None].expand(-1,-1,2,1)).squeeze(-1)
    half=size.gather(-1,other[...,None]).squeeze(-1)/2
    lateral=along-along.clamp(-half[...,None],half[...,None])
    vertical=hand_local[...,2]-hand_local[...,2].clamp(-size[...,None,2]/2,size[...,None,2]/2)
    distance=torch.stack((across,lateral,vertical),-1).norm(dim=-1)
    target=world_points(boxes,local)
    delta=hands.mean(-2)-target
    inward=torch.zeros_like(local)
    inward.scatter_(-1,axis[...,None],torch.where(side.remainder(2)==0,1.,-1.)[...,None])
    normal=rotate(boxes[...,3:7],inward)
    return target,delta,distance,normal


def face_contact(hands,forces,boxes,box_forces,size,push,side,local):
    _,_,distance,normal=point_geometry(hands,boxes,size,side,local)
    near=(distance<=push['hand_distance']).all(-1)
    reaction=-(forces*normal[:,:,None]).sum(-1)
    return near & (reaction>=push['min_hand_force']).all(-1) & ((box_forces*normal).sum(-1)>=push['min_box_force'])


def face_phi(hands,boxes,size,side,local):
    _,_,distance,_=point_geometry(hands,boxes,size,side,local)
    return torch.exp(-10*distance.amax(-1).square())


def validate_point_reward(env):
    """A deterministic material point is identifiable from the original input."""
    spec=env['pushPointReward']
    band='height_band_m' in spec
    if set(spec)!=({'height_band_m','z_reward_sigma_m'} if band else {'z_tolerance_m','z_reward_sigma_m'}):
        raise ValueError('Invalid point reward fields')
    sigma=spec['z_reward_sigma_m']
    tol=0. if band else spec['z_tolerance_m']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (tol,sigma)) or not 0<=tol<sigma or (not band and tol==0):
        raise ValueError('Invalid point reward height tolerance/sigma')
    lo,hi=env['pushContactPoint']['height_range_m']
    if lo!=hi:
        raise ValueError('Center-only observations require a fixed point height')
    if band:
        bounds=spec['height_band_m']
        if not isinstance(bounds,list) or len(bounds)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bounds):
            raise ValueError('Invalid point reward height band')
        if not 0<bounds[0]<bounds[1]<env['pushBox']['size'][2] or not math.isclose((bounds[0]+bounds[1])/2,lo,abs_tol=1e-9):
            raise ValueError('Height band must be centered on the fixed reference point')
    return spec


def point_height_excess(delta,spec):
    error=delta[...,2].abs()
    if 'height_band_m' in spec:
        lo,hi=spec['height_band_m']
        error=(error-(hi-lo)/2).clamp_min(0)
    return error


def point_height_valid(delta,spec):
    if 'height_band_m' in spec:
        lo,hi=spec['height_band_m']
        # Micrometer tolerance absorbs float32 subtraction at exact endpoints.
        return delta[...,2].abs()<=(hi-lo)/2+1e-6
    return delta[...,2].abs()<=spec['z_tolerance_m']


def point_reward_phi(hands,boxes,size,side,local,spec):
    _,delta,distance,_=point_geometry(hands,boxes,size,side,local)
    error=torch.maximum(delta[...,:2].norm(dim=-1),distance.amax(-1))
    return torch.exp(-10*error.square()-(point_height_excess(delta,spec)/spec['z_reward_sigma_m']).square())


def point_reward_contact(hands,forces,boxes,box_forces,size,push,side,local,spec):
    _,delta,_,_=point_geometry(hands,boxes,size,side,local)
    return (face_contact(hands,forces,boxes,box_forces,size,push,side,local)
            & (delta[...,:2].norm(dim=-1)<=push['hand_distance'])
            & point_height_valid(delta,spec))
