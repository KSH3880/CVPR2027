import torch
from env.tasks.multi_agent.joint_push_spawn import sample_rear_spawn


def test_independent_positions_heading_and_clearance():
    torch.manual_seed(123)
    depth,lateral,heading=sample_rear_spawn(2048,'cpu')
    assert ((depth>=1)&(depth<=2)).all()
    assert (lateral.abs()<=1.8).all()
    assert ((depth[:,0]-depth[:,1]).square()+(lateral[:,0]-lateral[:,1]).square()).sqrt().min()>=.8
    assert (lateral[:,0]*lateral[:,1]>0).any()
    assert (lateral[:,0]>lateral[:,1]).any() and (lateral[:,0]<lateral[:,1]).any()
    assert heading.min()<-3 and heading.max()>3
    assert (depth[:,0]-depth[:,1]).abs().mean()>.1


def test_surround_all_sides_clearance_and_independent_heading():
    from env.tasks.multi_agent.joint_push_spawn import sample_surround_spawn
    import math
    depth,lateral,heading=sample_surround_spawn(4096,'cpu',[2.6]*3)
    forward=-depth-1.3
    radius=(forward.square()+lateral.square()).sqrt()
    assert radius.min()>=math.hypot(2.6,2.6)/2+1-1e-6
    assert radius.max()<=math.hypot(2.6,2.6)/2+2+1e-6
    for x,y in ((1,1),(1,-1),(-1,1),(-1,-1)):
        assert ((forward*x>0)&(lateral*y>0)).any()
    gap=((forward[:,0]-forward[:,1]).square()+(lateral[:,0]-lateral[:,1]).square()).sqrt()
    assert gap.min()>=.8
    assert heading.min()<-3 and heading.max()>3
