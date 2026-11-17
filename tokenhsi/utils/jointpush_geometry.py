import torch


def yaw_from_quat(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))



def rotate(q,v):
    uv=torch.cross(q[...,:3],v,dim=-1)
    return v+2*(q[...,3:]*uv+torch.cross(q[...,:3],uv,dim=-1))



def multiply(a,b):
    av,bv=a[...,:3],b[...,:3]
    return torch.cat((a[...,3:]*bv+b[...,3:]*av+torch.cross(av,bv,dim=-1),
        a[...,3:]*b[...,3:]-(av*bv).sum(-1,keepdim=True)),-1)

def box_sizes(boxes, size):
    return torch.as_tensor(size, device=boxes.device, dtype=boxes.dtype).expand(*boxes.shape[:-1], 3)



def push_face_geometry(hands, boxes, goals, size):
    """Select one finite yaw-only face, within 60 degrees of the goal direction."""
    yaw=yaw_from_quat(boxes[...,3:7]);c,s=torch.cos(yaw),torch.sin(yaw)
    local_normals=boxes.new_tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    normals=torch.stack((c[...,None]*local_normals[:,0]-s[...,None]*local_normals[:,1],
                         s[...,None]*local_normals[:,0]+c[...,None]*local_normals[:,1]),-1)
    dims=box_sizes(boxes,size)/2
    half=dims[..., [0,0,1,1]]
    centers=boxes[...,:2].unsqueeze(-2)-normals*half[...,None]
    tangents=torch.stack((normals[...,1],-normals[...,0]),-1)
    widths=dims[..., [1,1,0,0]]
    delta=goals[...,:2]-boxes[...,:2]
    fallback=torch.stack((-s,c),-1)
    direction=torch.where((delta.norm(dim=-1)>1e-6)[...,None],
        delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6),fallback)
    allowed=(normals*direction[...,None,:]).sum(-1)>=.5-1e-6
    offset=hands[...,:2].unsqueeze(-3)-centers.unsqueeze(-2)
    along=(offset*tangents.unsqueeze(-2)).sum(-1)
    across=(offset*normals.unsqueeze(-2)).sum(-1)
    over=along-along.clamp(-widths[...,None],widths[...,None])
    worst=(over.square()+across.square()).amax(-1)
    face=worst.masked_fill(~allowed,float('inf')).argmin(-1)
    def select(value):
        return value.gather(-2,face[...,None,None].expand(*face.shape,1,2)).squeeze(-2)
    return select(centers),select(normals),select(tangents),widths.gather(-1,face[...,None]).squeeze(-1)



def push_rear_xy(hands, boxes, goals, size):
    center,normal,tangent,half=push_face_geometry(hands,boxes,goals,size)
    offset=hands[...,:2]-center.unsqueeze(-2)
    along=(offset*tangent.unsqueeze(-2)).sum(-1)
    across=(offset*normal.unsqueeze(-2)).sum(-1)
    over=along-along.clamp(-half[...,None],half[...,None])
    return (over.square()+across.square()).sqrt()



def push_goal_opposite_xy_target(boxes, goals, size):
    """Intersect the center-to-goal opposite ray with the yaw-only box rectangle."""
    yaw=yaw_from_quat(boxes[...,3:7])
    c,s=torch.cos(yaw),torch.sin(yaw)
    delta=goals[...,:2]-boxes[...,:2]
    local=torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)
    fallback=torch.zeros_like(local);fallback[...,1]=1.
    local=torch.where((local.norm(dim=-1)>1e-6)[...,None],local,fallback)
    half=box_sizes(boxes,size)[...,:2]/2
    rear=-local/(local.abs()/half).amax(-1,keepdim=True)
    return boxes[...,:2]+torch.stack((c*rear[...,0]-s*rear[...,1],s*rear[...,0]+c*rear[...,1]),-1)



def push_rsi_hand_targets(boxes, goals, size, config):
    """Align RSI hand midpoint and facing direction to the reward reference."""
    spec=config['push']
    if config.get('contact_at'):
        # RSI has no placed hands yet: use the opposite-goal ray as a seed.
        rear=push_goal_opposite_xy_target(boxes,goals,size)
        seed=torch.cat((rear,boxes[...,2:3]),-1).unsqueeze(-2).expand(*boxes.shape[:2],2,3)
        center,normal,tangent,_=push_face_geometry(seed,boxes,goals,size)
        targets=boxes.new_zeros(*boxes.shape[:2],2,3)
        targets[...,:2]=center.unsqueeze(-2)+tangent.unsqueeze(-2)*boxes.new_tensor(
            [spec['hand_lateral_offset'],-spec['hand_lateral_offset']])[...,None]
        targets[...,2]=(boxes[...,2]+spec['hand_target_height']-box_sizes(boxes,size)[...,2]/2)[...,None]
        return targets
    if config.get('push_at',{}).get('hand_reference')!='goal_opposite_surface_xy':
        return push_hand_targets(boxes,size,spec)
    rear=push_goal_opposite_xy_target(boxes,goals,size)
    normal=boxes[...,:2]-rear
    normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    right=torch.stack((normal[...,1],-normal[...,0]),-1)
    targets=boxes.new_zeros(*boxes.shape[:2],2,3)
    targets[...,:2]=rear.unsqueeze(-2)+right.unsqueeze(-2)*boxes.new_tensor(
        [spec['hand_lateral_offset'],-spec['hand_lateral_offset']])[...,None]
    targets[...,2]=spec['hand_target_height']
    return targets



def push_hand_targets(boxes, size, spec):
    """RSI right/left alignment points; also reward targets in the legacy mode."""
    offsets=boxes.new_tensor([[spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2],
                             [-spec['hand_lateral_offset'],-size[1]/2,spec['hand_target_height']-size[2]/2]])
    return boxes[...,:3][:,:,None]+rotate(boxes[...,3:7][:,:,None].expand(*boxes.shape[:2],2,4),
        offsets.expand(*boxes.shape[:2],2,3))
