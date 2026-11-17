"""Quaternion operations shared by scene placement and RSI."""
import torch

def rotate(q, v):
    cross = 2*torch.cross(q[..., :3].expand_as(v), v, dim=-1)
    return v+q[..., 3:]*cross+torch.cross(q[..., :3].expand_as(v), cross, dim=-1)
