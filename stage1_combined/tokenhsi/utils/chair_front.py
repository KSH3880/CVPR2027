"""Chair-front entry cone; central seating/standing region remains valid."""
import torch


def front_allowed(local_root, seat, facing, central=.12):
    f=facing/facing.norm(dim=-1,keepdim=True).clamp_min(1e-8);delta=local_root[...,:2]-seat[...,:2]
    forward=(delta*f).sum(-1);lateral=(delta*torch.stack([-f[...,1],f[...,0]],-1)).sum(-1)
    return (delta.norm(dim=-1)<=central)|((forward>=.03)&(lateral.abs()<=forward))
