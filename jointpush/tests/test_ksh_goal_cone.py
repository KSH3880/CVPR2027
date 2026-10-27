import math
import torch
from jointpush.ksh_goal_cone import yaw_goal_in_cone,goal_in_cone,sample_layout


def test_goal_cone_boundaries_and_rotated_box():
    yaw=torch.tensor([0.,math.pi/2,-math.pi+.01,math.pi-.01])[:,None]
    offsets=torch.deg2rad(torch.tensor([-180.,-46.,-45.,0.,45.,46.,180.]))[None]
    heading=yaw+math.pi/2+offsets
    delta=3*torch.stack((heading.cos(),heading.sin()),-1)
    expected=torch.tensor([False,False,True,True,True,False,False]).expand(4,-1)
    assert torch.equal(yaw_goal_in_cone(yaw,delta),expected)
    boxes=torch.zeros(4,7,13);boxes[...,5]=(yaw/2).sin();boxes[...,6]=(yaw/2).cos()
    goals=torch.zeros(4,7,3);goals[...,:2]=delta
    assert torch.equal(goal_in_cone(boxes,goals),expected)


def test_stage1_resamples_to_cone_without_changing_distance_ranges():
    torch.manual_seed(42)
    roots=torch.zeros(64,2,3);roots[:,0,0]=-15;roots[:,1,0]=15
    boxes=torch.zeros(64,4,13);boxes[...,6]=1
    sizes=torch.full((64,4,3),1.6)
    push=torch.ones(64,2,dtype=torch.bool)
    spec=dict(source_distance=[2.,10.],goal_distance=[2.,10.],rear_clearance=1.5,path_separation=3.5,agent_clearance=.5,obstacle_clearance=.5,goal_min_human_distance=1.)
    selected,goals=sample_layout(roots,boxes,sizes,push,spec)
    assert goal_in_cone(selected[:,:2],goals).all()
    source_distance=(selected[:,:2,:2]-roots[...,:2]).norm(dim=-1)
    target_distance=(goals[...,:2]-selected[:,:2,:2]).norm(dim=-1)
    assert ((source_distance>=2-1e-5)&(source_distance<=10+1e-5)).all()
    assert ((target_distance>=2-1e-5)&(target_distance<=10+1e-5)).all()


def test_stage1_aligned_goal_has_zero_lateral_offset():
    torch.manual_seed(42)
    roots=torch.zeros(64,2,3);roots[:,0,0]=-15;roots[:,1,0]=15
    boxes=torch.zeros(64,4,13);boxes[...,6]=1
    sizes=torch.full((64,4,3),1.6)
    spec=dict(source_distance=[2.,10.],goal_distance=[2.,10.],rear_clearance=1.5,path_separation=3.5,agent_clearance=.5,obstacle_clearance=.5,goal_min_human_distance=1.,goal_half_angle_degrees=0.)
    selected,goals=sample_layout(roots,boxes,sizes,torch.ones(64,2,dtype=torch.bool),spec)
    from utils.stage2_jointpush import inverse_rotate
    local=inverse_rotate(selected[:,:2,3:7],goals-selected[:,:2,:3])
    assert local[...,0].abs().max()<1e-5
    assert ((local[...,1]>=2-1e-5)&(local[...,1]<=10+1e-5)).all()
