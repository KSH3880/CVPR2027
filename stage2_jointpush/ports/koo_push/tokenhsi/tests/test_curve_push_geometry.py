import math
import torch
from env.tasks.multi_agent.curve_push_geometry import sample_curve_goal_offset,fixed_rear_geometry
from env.tasks.multi_agent.joint_push_reward import DEFAULT_CONFIG,joint_push_reward


def test_initial_goal_misalignment_and_distance():
    torch.manual_seed(42)
    yaw=torch.linspace(-math.pi,math.pi,1024)
    distance=torch.linspace(2.2,2.8,1024)
    offset,angle=sample_curve_goal_offset(yaw,distance)
    x=yaw.cos()*offset[:,0]+yaw.sin()*offset[:,1]
    y=-yaw.sin()*offset[:,0]+yaw.cos()*offset[:,1]
    assert torch.allclose(offset[:,:2].norm(dim=-1),distance,atol=1e-6)
    assert torch.allclose(torch.atan2(-x,y),angle,atol=1e-6)
    assert (angle.abs()>=math.radians(20)).all() and (angle.abs()<=math.radians(30)).all()
    assert (angle<0).any() and (angle>0).any() and (y.abs()>x.abs()).all()


def test_contact_face_stays_on_box_after_turn_and_goal_passes_side():
    box=torch.zeros(1,13);box[:,2]=1.3;box[:,6]=1.
    geometry=fixed_rear_geometry(box,[2.6]*3,DEFAULT_CONFIG)
    assert torch.allclose(geometry[0],torch.tensor([[0.,-1.3,.8]]))
    hands=geometry[0][:,None,None,:].expand(1,2,2,3).clone()
    hands[:,0,:,0]=-.65;hands[:,1,:,0]=.65
    roots=torch.zeros(1,2,13)
    r=joint_push_reward(roots,hands,box,torch.tensor([[3.,0.,1.3]]),[2.6]*3,face_geometry=geometry)
    assert r['coupled_contact'].all()
    box[:,5:7]=2**-.5
    geometry=fixed_rear_geometry(box,[2.6]*3,DEFAULT_CONFIG)
    assert torch.allclose(geometry[0],torch.tensor([[1.3,0.,.8]]),atol=1e-6)
    assert torch.allclose(geometry[1],torch.tensor([[-1.,0.]]),atol=1e-6)
