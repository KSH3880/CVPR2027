import math
import torch
import pytest
from env.tasks.multi_agent.joint_push_reward import DEFAULT_CONFIG,joint_push_reward


def scene():
    b=torch.zeros(1,13);b[:,2]=.75;b[:,6]=1
    r=torch.zeros(1,2,13);r[...,6]=1;r[0,0,:2]=torch.tensor([-2.,0.]);r[0,1,:2]=torch.tensor([0.,-2.])
    h=torch.tensor([[[[-1.3,-.15,.8],[-1.3,.15,.8]],[[-.15,-1.3,.8],[.15,-1.3,.8]]]])
    g=torch.tensor([[3.,3.,.75]])
    cfg=dict(DEFAULT_CONFIG,contact_mode='directional_surface',at_gate_mode='soft_min')
    return r,h,b,g,cfg


def reward(r,h,b,g,cfg):return joint_push_reward(r,h,b,g,[2.6,2.6,1.5],cfg)


def test_adjacent_faces_diagonal_and_agent_permutation():
    r,h,b,g,c=scene();out=reward(r,h,b,g,c)
    assert out['coupled_contact'].all()
    torch.testing.assert_close(out['phi_raw'][...,0],torch.ones(1,2))
    assert out['dense_at_gate'].item()==1
    other=reward(r.flip(1),h.flip(1),b,g,c)
    torch.testing.assert_close(out['total'].flip(1),other['total'])


def test_wrong_face_and_hollow_midpoint_rejected():
    r,h,b,g,c=scene();h[:,0,:,0]=1.3
    assert reward(r,h,b,g,c)['phi_raw'][0,0,0]==0
    r,h,b,g,c=scene();h[:,0,0,0]=-2.;h[:,0,1,0]=-.6
    assert reward(r,h,b,g,c)['phi_raw'][0,0,0]<.1


def test_height_and_separation():
    r,h,b,g,c=scene();h[...,2]=1.3
    assert not reward(r,h,b,g,c)['coupled_contact'].any()
    r,h,b,g,c=scene();h[:,1]=h[:,0]
    assert reward(r,h,b,g,c)['dense_at_gate'].item()==0


def test_yaw_rotation_invariance_and_rectangle_progress():
    r,h,b,g,c=scene();before=reward(r,h,b,g,c)
    angle=.63;co=math.cos(angle);si=math.sin(angle)
    for a in (r,h,b,g):
        xy=a[...,:2].clone();a[...,0]=co*xy[...,0]-si*xy[...,1];a[...,1]=si*xy[...,0]+co*xy[...,1]
    b[:,5]=math.sin(angle/2);b[:,6]=math.cos(angle/2)
    after=reward(r,h,b,g,c)
    torch.testing.assert_close(before['total'],after['total'])
    # Root at x=-2 is .7m from the actual face: .2m buffer -> 1/1.5.
    assert before['progress_raw'][0,0,0].item()==pytest.approx(1/1.5)


def test_at_distance_formula_preserved_and_new_contact_gate():
    r,h,b,g,c=scene();g[0]=torch.tensor([.05,.05,.75])
    out=reward(r,h,b,g,c)
    assert out['done'].all()
    assert out['phi_raw'][0,0,1].item()==pytest.approx(math.exp(-10*.005))
    h[:,0,:,0]=1.3
    assert not reward(r,h,b,g,c)['done'].any()
