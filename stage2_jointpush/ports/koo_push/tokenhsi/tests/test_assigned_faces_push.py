import math
from pathlib import Path
import pytest
import torch
import yaml
from env.tasks.multi_agent.joint_push_assigned_contact import assign_contact_targets,world_targets,rotate
from env.tasks.multi_agent.joint_push_reward import DEFAULT_CONFIG,joint_push_reward

SPEC=dict(split_angle_deg=25.,hand_spacing=.3,same_face_offset_fraction=.25,approach_standoff=.8)
CFG=dict(DEFAULT_CONFIG,contact_mode='assigned_faces',at_gate_mode='soft_min')


def scene(angle=0.,count=1):
    b=torch.zeros(count,13);b[:,2]=1.;b[:,6]=1
    sizes=torch.tensor([[2.6,2.6,2.]]).expand(count,-1).clone()
    g=b[:,:3].clone();g[:,0]=3*math.sin(math.radians(angle));g[:,1]=3*math.cos(math.radians(angle))
    r=torch.zeros(count,2,13);r[...,6]=1;r[:,0,:2]=torch.tensor([-.8,-2.]);r[:,1,:2]=torch.tensor([.8,-2.])
    a=assign_contact_targets(b,g,sizes,r,SPEC)
    return b,g,sizes,r,a


def evaluate(b,g,sizes,r,a,h=None):
    ht=world_targets(b,a['local_hands']);ap=world_targets(b,a['local_approach'])
    return joint_push_reward(r,ht if h is None else h,b,g,sizes,CFG,assigned_targets=(ht,ap))


@pytest.mark.parametrize('angle,two',[(0,False),(24.9,False),(25,False),(25.1,True),(45,True),(90,False),(135,True),(180,False),(-45,True),(-135,True)])
def test_angle_face_selection_and_finite_hand_targets(angle,two):
    b,g,s,r,a=scene(angle)
    assert a['two_faces'].item()==two
    dot=(a['local_normals'][:,0]*a['local_normals'][:,1]).sum(-1)
    assert dot.item()==(0 if two else 1)
    assert (a['local_hands'][...,:2].abs()<=s[:,None,None,:2]/2+1e-6).all()
    torch.testing.assert_close(world_targets(b,a['local_hands'])[...,2],torch.full((1,2,2),.8))
    assert evaluate(b,g,s,r,a)['coupled_contact'].all()


def test_agent_assignment_minimizes_distance_and_is_permutation_equivariant():
    b,g,s,r,a=scene(45)
    swapped=assign_contact_targets(b,g,s,r.flip(1),SPEC)
    torch.testing.assert_close(a['local_hands'].flip(1),swapped['local_hands'])
    ap=world_targets(b,a['local_approach'])
    assert (r[...,:2]-ap[...,:2]).norm(dim=-1).sum()<=(r[...,:2]-ap.flip(1)[...,:2]).norm(dim=-1).sum()


def test_episode_fixed_targets_follow_translation_and_rotation_not_goal():
    b,g,s,r,a=scene(24)
    saved=a['local_hands'].clone()
    # Goal crossing the angular boundary must not reassign an existing episode.
    g[:,0]=3.;g[:,1]=3.
    assert evaluate(b,g,s,r,a)['coupled_contact'].all()
    assert not a['two_faces'].item()
    yaw=.9;b[:,5]=math.sin(yaw/2);b[:,6]=math.cos(yaw/2);b[:,:2]=torch.tensor([[4.,-2.]])
    expected=rotate(b[:,3:7],saved)+b[:,None,None,:3]
    torch.testing.assert_close(world_targets(b,saved),expected)
    assert evaluate(b,g,s,r,a)['coupled_contact'].all()
    torch.testing.assert_close(saved,a['local_hands'])


def test_wrong_face_progress_and_high_hands_rejected():
    b,g,s,r,a=scene()
    ap=world_targets(b,a['local_approach'])
    r[...,:2]=ap[...,:2]
    good=evaluate(b,g,s,r,a)
    torch.testing.assert_close(good['progress_raw'][...,0],torch.ones(1,2))
    r[...,1]*=-1
    bad=evaluate(b,g,s,r,a)
    assert (bad['progress_raw'][...,0]<.3).all()
    hands=world_targets(b,a['local_hands']);hands[...,2]=2.
    assert not evaluate(b,g,s,r,a,hands)['own_success'][...,0].any()
    hands=world_targets(b,a['local_hands']);hands[:,0,0,0]+=.4
    assert not evaluate(b,g,s,r,a,hands)['own_success'][0,0,0]


def test_at_preserved_and_reset_subset_buffer_update():
    b,g,s,r,a=scene(45,count=3)
    original=a['local_hands'].clone()
    b2,g2,s2,r2,new=scene(0)
    for key in a:a[key][1:2]=new[key]
    torch.testing.assert_close(a['local_hands'][[0,2]],original[[0,2]])
    g=b[:,:3]+torch.tensor([.05,.05,0.])
    out=evaluate(b,g,s,r,a)
    assert out['done'].all()
    assert out['phi_raw'][0,0,1].item()==pytest.approx(math.exp(-.05))


def test_profile_height_grid_and_hand_order():
    cfg=yaml.safe_load((Path(__file__).parents[1]/'data/cfg/multi_agent/joint_push_assigned_faces_ca4.yaml').read_text())['env']
    from env.tasks.multi_agent.joint_push_shape import shape_mass_grid
    grid=shape_mass_grid(cfg['jointPush']['shape_randomization'])
    assert len(grid)==1620
    assert min(x[0][2] for x in grid)==1.6 and max(x[0][2] for x in grid)==2.6
    assert cfg['keyBodies'][:2]==['right_hand','left_hand']
    assert cfg['relationReward']['joint_push_reward']==cfg['jointPush']['reward']


def test_world_target_transform_known_quaternion_and_missing_assignment():
    b,g,s,r,a=scene()
    local=a['local_hands'].clone()
    b[:,:3]=torch.tensor([[4.,5.,1.]])
    b[:,5]=math.sqrt(.5);b[:,6]=math.sqrt(.5)
    expected=local.clone();expected[...,0]=4-local[...,1];expected[...,1]=5+local[...,0];expected[...,2]=1+local[...,2]
    torch.testing.assert_close(world_targets(b,local),expected)
    with pytest.raises(ValueError,match='reset-time'):
        joint_push_reward(r,world_targets(b,local),b,g,s,CFG)
