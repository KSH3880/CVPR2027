import torch
from .reward import rotate

def multiply(a,b):
    xyz=a[...,3:]*b[...,:3]+b[...,3:]*a[...,:3]+torch.cross(a[...,:3],b[...,:3],dim=-1)
    return torch.cat((xyz,a[...,3:]*b[...,3:]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)),-1)

def yaw_from_quat(q):
    return torch.atan2(2*(q[...,3]*q[...,2]+q[...,0]*q[...,1]),1-2*(q[...,1].square()+q[...,2].square()))
