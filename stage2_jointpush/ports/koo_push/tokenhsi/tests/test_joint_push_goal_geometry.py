import math
import torch
from env.tasks.multi_agent.joint_push_goal_geometry import aligned_goal_offset
from env.tasks.multi_agent.joint_push_reward import push_face_geometry,DEFAULT_CONFIG


def test_aligned_goals_match_actual_face_across_all_rotations():
    yaw=torch.linspace(-math.pi,math.pi,1024)
    distance=torch.linspace(2.2,2.8,1024)
    offsets=aligned_goal_offset(yaw,distance)
    box=torch.zeros(1024,13);box[:,2]=1.3;box[:,5]=(yaw/2).sin();box[:,6]=(yaw/2).cos()
    goal=box[:,:3]+offsets
    center,normal,_,_,_=push_face_geometry(box,goal,[2.6]*3,DEFAULT_CONFIG)
    assert torch.allclose(offsets[:,:2].norm(dim=-1),distance,atol=1e-6)
    assert torch.allclose(offsets[:,:2]/distance[:,None],normal,atol=1e-6)
    assert torch.allclose(center[:,:2],-1.3*normal,atol=1e-6)


def test_aligned_goal_rotates_with_box_in_world():
    yaw=torch.tensor([-.5,.2,2.3]);turn=.7;distance=torch.tensor([2.2,2.5,2.8])
    original=aligned_goal_offset(yaw,distance)
    rotated=torch.stack((math.cos(turn)*original[:,0]-math.sin(turn)*original[:,1],math.sin(turn)*original[:,0]+math.cos(turn)*original[:,1],original[:,2]),-1)
    assert torch.allclose(aligned_goal_offset(yaw+turn,distance),rotated,atol=1e-6)
