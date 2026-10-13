"""Misaligned goal sampling and fixed rear-face geometry for curved Push probes."""
import math
import torch
from env.tasks.multi_agent.joint_push_reward import push_face_geometry


def sample_curve_goal_offset(yaw, distance, angle_abs_deg=(20.,30.)):
    lo,hi=angle_abs_deg
    if not 0<lo<=hi<45:raise ValueError('Curve misalignment must be between 0 and 45 degrees')
    angle=(lo+(hi-lo)*torch.rand_like(yaw))*math.pi/180
    angle=angle*torch.where(torch.rand_like(yaw)<.5,-1.,1.)
    heading=yaw+math.pi/2+angle
    offset=torch.stack((distance*heading.cos(),distance*heading.sin(),torch.zeros_like(distance)),-1)
    return offset,angle


def fixed_rear_geometry(boxes,sizes,config):
    x,y,z,w=boxes[:,3:7].unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    # Always the original local -Y face, regardless of where the remaining goal lies.
    target=boxes[:,:3].clone()
    target[:,0]-=yaw.sin();target[:,1]+=yaw.cos()
    return push_face_geometry(boxes,target,sizes,config)
