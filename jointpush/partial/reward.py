"""RawCuboid contact/AT shaping, adapted only for regions and shared physical boxes."""
import torch


def rotate(q, v):
    cross = 2*torch.cross(q[..., :3], v, dim=-1)
    return v+q[..., 3:]*cross+torch.cross(q[..., :3], cross, dim=-1)


def inverse_rotate(q, v):
    return rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), v)


def approach_distance(roots, centers, normals, half_size, target_x, stand_off=1., clearance=.45):
    """Baseline obstacle-aware approach route, ending at the assigned rear region."""
    delta=roots[..., :2]-centers[..., :2]
    tangent=torch.stack((normals[..., 1],-normals[..., 0]),-1)
    start=torch.stack(((delta*tangent).sum(-1),(delta*normals[..., :2]).sum(-1)),-1)
    extent=half_size[..., :2]+clearance
    target=torch.stack((target_x,-half_size[..., 1]-stand_off),-1)
    left=torch.stack((-extent[..., 0],-extent[..., 1]),-1)
    right=torch.stack((extent[..., 0],-extent[..., 1]),-1)
    back_left=left.clone();back_left[...,1]*=-1
    back_right=right.clone();back_right[...,1]*=-1
    destinations=torch.stack((target,left,right,back_left,back_right),-2)
    remaining=torch.stack((torch.zeros_like(target_x),(left-target).norm(dim=-1),(right-target).norm(dim=-1),2*extent[...,1]+(left-target).norm(dim=-1),2*extent[...,1]+(right-target).norm(dim=-1)),-1)
    segment=destinations-start[...,None,:]
    parallel=segment.abs()<1e-7
    denominator=torch.where(parallel,torch.ones_like(segment),segment)
    lo=(-extent[...,None,:]+1e-4-start[...,None,:])/denominator
    hi=(extent[...,None,:]-1e-4-start[...,None,:])/denominator
    inside=start[...,None,:].abs()<extent[...,None,:]-1e-4
    enter=torch.where(parallel,torch.where(inside,-torch.inf,torch.inf),torch.minimum(lo,hi)).amax(-1)
    leave=torch.where(parallel,torch.where(inside,torch.inf,-torch.inf),torch.maximum(lo,hi)).amin(-1)
    blocked=torch.maximum(enter,torch.zeros_like(enter))<torch.minimum(leave,torch.ones_like(leave))
    paths=(segment.norm(dim=-1)+remaining).masked_fill(blocked,torch.inf)
    distance=paths.amin(-1)
    return torch.where(torch.isfinite(distance),distance,(start-target).norm(dim=-1))


def components(roots,hands,hand_force,boxes,sizes,box_force,assignment,local_points,goals,spec,contact_geometry,hits=None):
    rows=torch.arange(len(roots),device=roots.device)[:,None]
    box=boxes[rows,assignment];size=sizes[rows,assignment]
    same=assignment[...,None]==assignment[:,None,:]
    counts=same.sum(-1)
    # The region is a policy/contact target; the goal always uses the physical box.
    region_box=box.clone();offset=torch.zeros_like(box[...,:3]);offset[...,0]=local_points[...,0].mean(-1)
    region_box[...,:3]+=rotate(box[...,3:7],offset)
    region_size=size.clone();region_size[...,0]/=counts
    phi_contact,hit,point,normal=contact_geometry(hands,region_box,region_size,hand_force,box_force[rows,assignment],midpoint_only=spec['contact_midpoint_only'],contact_height=spec['contact_height'])
    if hits is not None:hit=hits
    delta=box[...,:3]-goals
    phi_at=torch.exp(-10*delta.square().sum(-1))
    local=inverse_rotate(box[...,None,3:7].expand(*hands.shape[:-1],4),hands-box[...,None,:3])
    excess_x=(local[...,0].abs()-size[...,None,0]/2).clamp_min(0)
    excess_z=(local[...,2].abs()-size[...,None,2]/2).clamp_min(0)
    face_distance=(excess_x.square()+excess_z.square()+(local[...,1]+size[...,None,1]/2).square()).sqrt()
    behind=(local[...,1]<=0).all(-1)
    goal_vector=goals[...,:2]-box[...,:2]
    aligned=(normal[...,:2]*goal_vector).sum(-1)>0
    at_position=(phi_at>=.9)&(delta[...,2].abs()<=.001)
    face=(face_distance<=spec['face_tolerance']).all(-1)&behind&(aligned|at_position)
    gate=(face[:,None,:]|~same).all(-1)
    at=at_position&gate
    saturated=hit|at
    distance=approach_distance(roots,box[...,:3],normal,size/2,offset[...,0],spec['approach_stand_off'],spec['clearance'])
    progress_contact=1/(1+distance)
    progress_at=1/(1+(delta[...,:2].norm(dim=-1)-.1).clamp_min(0))
    state=torch.stack((torch.where(saturated,1.,phi_contact),torch.where(at,1.,phi_at)*gate),-1)*.2
    progress=torch.stack((torch.where(saturated,1.,progress_contact),torch.where(at,1.,progress_at)*gate),-1)*.2
    success=torch.stack((saturated,at),-1).float()*.2
    local_reward=(state+progress+success).sum(-1)
    other=(same.float()*local_reward[:,None,:]).sum(-1)-local_reward
    other=torch.where(counts>1,other/(counts-1).clamp_min(1),local_reward)
    reward=.9*local_reward+.1*other
    return dict(reward=reward,state=state,progress=progress,success=success,phi=torch.stack((phi_contact,phi_at),-1),contact=hit,done=at,group_gate=gate,hand_face=face,face_distance=face_distance,goal_distance=delta[...,:2].norm(dim=-1),raw_at_progress=progress_at,targets=point,normal=normal)
