"""Pure geometry for two rear-aligned Stage1-sized observation boxes of a shared rigid box."""
import torch


def split_box_observation(boxes, sizes, goals, split_axis, forward_sign):
    sizes=torch.as_tensor(sizes,device=boxes.device,dtype=boxes.dtype).expand(len(boxes),3)
    axis=split_axis.long()
    virtual_sizes=boxes.new_full((len(boxes),2,3),1.3)
    if (sizes<1.3).any():raise ValueError('Physical box must contain the 1.3m virtual cubes')
    offsets=boxes.new_zeros(len(boxes),2,3)
    batch=torch.arange(len(boxes),device=boxes.device)
    normal_axis=1-axis
    offsets[batch,0,axis]=-(sizes[batch,axis]-1.3)/2
    offsets[batch,1,axis]=(sizes[batch,axis]-1.3)/2
    # Match the actual rear face and bottom, rather than the actual box center.
    offsets[batch,:,normal_axis]=-forward_sign[:,None]*(sizes[batch,normal_axis,None]-1.3)/2
    offsets[:,:,2]=-(sizes[:,2,None]-1.3)/2
    # Local lateral offset rotates with the actual rigid box (including tilt).
    q=boxes[:,None,3:7].expand(-1,2,-1)
    xyz=q[...,:3]
    t=2*torch.cross(xyz,offsets,dim=-1)
    world=offsets+q[...,3:4]*t+torch.cross(xyz,t,dim=-1)
    states=boxes[:,None].expand(-1,2,-1).clone()
    states[...,:3]+=world
    states[...,7:10]+=torch.cross(states[...,10:13],world,dim=-1)
    signs=boxes.new_tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],[1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]])
    corners=virtual_sizes[:,:,None]/2*signs
    targets=goals[:,None]+world
    return states,corners,targets,virtual_sizes
