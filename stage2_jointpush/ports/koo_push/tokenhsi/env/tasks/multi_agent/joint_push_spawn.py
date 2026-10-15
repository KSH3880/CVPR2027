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


def sample_surround_spawn(count, device, size, clearance=(1.,2.), minimum_gap=.8):
    """Uniform azimuth outside the XY circumscribed circle; independent headings."""
    if len(size)!=3 or min(size)<=0 or not 0<clearance[0]<=clearance[1]:
        raise ValueError('Invalid surround spawn geometry')
    bound=math.hypot(size[0],size[1])/2
    def candidates(n):
        angle=torch.empty(n,2,device=device).uniform_(-math.pi,math.pi)
        radius=bound+torch.empty_like(angle).uniform_(*clearance)
        return radius*angle.sin(),radius*angle.cos()
    forward,lateral=candidates(count)
    for _ in range(128):
        bad=((forward[:,0]-forward[:,1]).square()+(lateral[:,0]-lateral[:,1]).square()).sqrt()<minimum_gap
        if not bad.any():break
        forward[bad],lateral[bad]=candidates(int(bad.sum()))
    else:raise RuntimeError('Could not sample separated surrounding positions')
    depth=-forward-size[1]/2
    heading=torch.empty_like(depth).uniform_(-math.pi,math.pi)
    return depth,lateral,heading
