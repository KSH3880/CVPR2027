"""Independent rear-side starting locations and headings for shared Push."""
import math
import torch


def sample_rear_spawn(count, device, depth_range=(1.,2.), lateral_limit=1.8, minimum_gap=.8):
    if not 0 < depth_range[0] <= depth_range[1] or not 0 < minimum_gap < 2*lateral_limit:
        raise ValueError('Invalid rear spawn bounds')
    depth=torch.empty(count,2,device=device).uniform_(*depth_range)
    lateral=torch.empty_like(depth).uniform_(-lateral_limit,lateral_limit)
    for _ in range(128):
        bad=((depth[:,0]-depth[:,1]).square()+(lateral[:,0]-lateral[:,1]).square()).sqrt()<minimum_gap
        if not bad.any():break
        depth[bad]=torch.empty(int(bad.sum()),2,device=device).uniform_(*depth_range)
        lateral[bad]=torch.empty(int(bad.sum()),2,device=device).uniform_(-lateral_limit,lateral_limit)
    else:raise RuntimeError('Could not sample separated starting positions')
    heading=torch.empty_like(depth).uniform_(-math.pi,math.pi)
    return depth,lateral,heading


def sample_surround_spawn(count, device, size, clearance=(1.,2.), minimum_gap=.8, num_agents=2):
    """Uniform azimuth outside the XY circumscribed circle; independent headings."""
    if torch.is_tensor(size) and size.ndim==2:
        if size.shape!=(count,3) or not (size>0).all():raise ValueError('Expected positive size [N,3]')
        bound=size[:,:2].norm(dim=-1)[:,None]/2
        half_depth=size[:,1,None]/2
    else:
        if len(size)!=3 or min(size)<=0:raise ValueError('Invalid surround spawn geometry')
        bound=math.hypot(size[0],size[1])/2
        half_depth=size[1]/2
    if not 0<clearance[0]<=clearance[1]:raise ValueError('Invalid spawn clearance')
    def candidates(n,rows=None):
        angle=torch.empty(n,num_agents,device=device).uniform_(-math.pi,math.pi)
        limits=bound[rows] if torch.is_tensor(bound) and rows is not None else bound
        radius=limits+torch.empty_like(angle).uniform_(*clearance)
        return radius*angle.sin(),radius*angle.cos()
    forward,lateral=candidates(count)
    for _ in range(128):
        positions=torch.stack((forward,lateral),-1)
        distances=torch.cdist(positions,positions)
        distances.diagonal(dim1=-2,dim2=-1).fill_(float('inf'))
        bad=distances.amin(dim=(-2,-1))<minimum_gap
        if not bad.any():break
        forward[bad],lateral[bad]=candidates(int(bad.sum()),bad)
    else:raise RuntimeError('Could not sample separated surrounding positions')
    depth=-forward-half_depth
    heading=torch.empty_like(depth).uniform_(-math.pi,math.pi)
    return depth,lateral,heading
