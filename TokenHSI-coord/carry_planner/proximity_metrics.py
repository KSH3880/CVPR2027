"""3D proximity proxies; body centers and oriented box surfaces, no contact claim."""
import itertools
import math
import torch

KINDS = ('agent_agent', 'agent_box', 'box_box', 'total')


def rotation_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def point_box_distance(points, center, rotation, half):
    local = torch.einsum('...ni,...ij->...nj', points-center[..., None, :], rotation)
    return (local.abs()-half[..., None, :]).clamp_min(0).norm(dim=-1)


def box_box_distance(center, rotation, half):
    signs = torch.tensor(list(itertools.product((-1., 1.), repeat=3)), device=center.device, dtype=center.dtype)
    vertices = torch.einsum('eani,eaji->eanj', half[:, :, None, :]*signs, rotation) + center[:, :, None, :]
    a, b = vertices[:, 0], vertices[:, 1]
    distance = torch.minimum(
        point_box_distance(a, center[:, 1], rotation[:, 1], half[:, 1]).amin(-1),
        point_box_distance(b, center[:, 0], rotation[:, 0], half[:, 0]).amin(-1))
    edges = [(i, j) for i in range(8) for j in range(i+1, 8) if (i ^ j) in (1, 2, 4)]
    # All edge pairs: endpoints plus the interior/interior stationary point.
    ix = torch.tensor(edges, device=center.device)
    p = a[:, ix[:, 0]][:, :, None]; u = (a[:, ix[:, 1]]-a[:, ix[:, 0]])[:, :, None]
    q = b[:, ix[:, 0]][:, None]; v = (b[:, ix[:, 1]]-b[:, ix[:, 0]])[:, None]
    r = p-q
    uu, vv, uv = (u*u).sum(-1), (v*v).sum(-1), (u*v).sum(-1)
    ur, vr = (u*r).sum(-1), (v*r).sum(-1)
    denom = uu*vv-uv*uv
    s = (uv*vr-vv*ur)/denom.clamp_min(1e-12)
    t = (uu*vr-uv*ur)/denom.clamp_min(1e-12)
    valid = (denom > 1e-12) & (s>=0) & (s<=1) & (t>=0) & (t<=1)
    interior = (r+s[...,None]*u-t[...,None]*v).norm(dim=-1).masked_fill(~valid, float('inf'))
    options = [interior]
    for endpoint in (0., 1.):
        rr = r+endpoint*u
        tt = ((rr*v).sum(-1)/vv.clamp_min(1e-12)).clamp(0,1)
        options.append((rr-tt[...,None]*v).norm(dim=-1))
        rr = r-endpoint*v
        ss = (-(rr*u).sum(-1)/uu.clamp_min(1e-12)).clamp(0,1)
        options.append((rr+ss[...,None]*u).norm(dim=-1))
    distance = torch.minimum(distance, torch.stack(options).amin(0).flatten(1).amin(-1))
    # SAT also catches intersecting boxes whose vertices are all outside.
    axes_a, axes_b = rotation[:,0].transpose(-1,-2), rotation[:,1].transpose(-1,-2)
    cross = torch.cross(axes_a[:,:,None].expand(-1,3,3,-1), axes_b[:,None].expand(-1,3,3,-1),dim=-1).flatten(1,2)
    axes = torch.cat((axes_a,axes_b,cross),1)
    lengths = axes.norm(dim=-1)
    axes = axes/lengths.clamp_min(1e-12)[...,None]
    radii_a = (torch.einsum('eki,eij->ekj',axes,rotation[:,0]).abs()*half[:,0,None]).sum(-1)
    radii_b = (torch.einsum('eki,eij->ekj',axes,rotation[:,1]).abs()*half[:,1,None]).sum(-1)
    separation = (axes*(center[:,1]-center[:,0])[:,None]).sum(-1).abs()-radii_a-radii_b
    disjoint = ((separation>1e-7)&(lengths>1e-7)).any(-1)
    return torch.where(disjoint,distance,torch.zeros_like(distance))


def proximity_distances(body, boxes, sizes):
    """body [E,2,B,3], box roots [E,2,13], sizes [E,2,3]."""
    rotation = rotation_matrix(boxes[...,3:7]); half=sizes*0.5
    aa = torch.cdist(body[:,0],body[:,1]).flatten(1).amin(-1)
    ab = torch.minimum(
        point_box_distance(body[:,0],boxes[:,1,:3],rotation[:,1],half[:,1]).amin(-1),
        point_box_distance(body[:,1],boxes[:,0,:3],rotation[:,0],half[:,0]).amin(-1))
    bb = box_box_distance(boxes[...,:3],rotation,half)
    return dict(agent_agent=aa,agent_box=ab,box_box=bb)


class ProximityTracker:
    def __init__(self, envs, device, threshold=.3):
        if not math.isfinite(threshold) or threshold<=0: raise ValueError('threshold must be positive and finite')
        self.threshold=threshold
        self.steps=torch.zeros(envs,4,dtype=torch.long,device=device)
        self.minimum=torch.full((envs,3),float('inf'),device=device)
    def update(self, distances):
        d=torch.stack([distances[k] for k in KINDS[:3]],-1)
        hits=d<self.threshold
        self.steps+=torch.cat((hits,hits.any(-1,keepdim=True)),-1).long()
        self.minimum=torch.minimum(self.minimum,d)
    def reset(self, ids):
        self.steps[ids]=0;self.minimum[ids]=float('inf')
