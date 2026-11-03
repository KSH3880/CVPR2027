from copy import deepcopy
import pytest
import torch
from utils.premature_push import motion_penalty, validate_config
from utils.experiments import native_config, validate_experiment
from utils.coupled_push_spec import checkpoint_contract, check_checkpoint


def scene(groups=(0,0,0,0)):
    m=len(groups)
    boxes=torch.zeros(1,m,13); boxes[...,6]=1; boxes[...,8]=1
    sizes=torch.ones(1,m,3)*2
    hands=torch.zeros(1,m,2,3); hands[...,1]=-1
    hands[:, -1, :, 1]=-2  # one teammate has not reached the box
    forces=torch.zeros_like(hands)
    return dict(hands=hands,hand_forces=forces,boxes=boxes,sizes=sizes,
                groups=torch.tensor([groups]),templates=torch.full((1,m),5),
                tolerance=.18,weight=.2,speed_threshold=.05,force_epsilon=1e-6)


def test_only_pusher_pays_and_stationary_waiter_is_zero():
    s=scene(); s['hand_forces'][0,0,:,1]=-10
    v=motion_penalty(**s)
    torch.testing.assert_close(v['penalty'],torch.tensor([[-.19,0,0,0]]))
    torch.testing.assert_close(v['share'],torch.tensor([[1.,0,0,0]]))


def test_force_ratio_conserves_one_group_cost():
    s=scene(); s['hand_forces'][0,0,:,1]=-10; s['hand_forces'][0,1,:,1]=-30
    v=motion_penalty(**s)
    torch.testing.assert_close(v['penalty'],torch.tensor([[-.0475,-.1425,0,0]]))
    assert v['penalty'].sum().item()==pytest.approx(-.19)


@pytest.mark.parametrize('case',['ready','stationary','coasting','braking','nonpush','solo','far_force','height','zero_weight'])
def test_no_penalty_cases(case):
    s=scene(); s['hand_forces'][0,0,:,1]=-10
    if case=='ready':s['hands'][... ,1]=-1
    if case=='stationary':s['boxes'][...,8]=.04
    if case=='coasting':s['hand_forces'].zero_()
    if case=='braking':s['hand_forces'][0,0,:,1]=10
    if case=='nonpush':s['templates'].fill_(3)
    if case=='solo':s['groups']=torch.arange(4)[None]
    if case=='far_force':s['hands'][0,0,:,1]=-2
    if case=='height':s['hands'][0,0,:,2]=4
    if case=='zero_weight':s['weight']=0.
    assert torch.count_nonzero(motion_penalty(**s)['penalty'])==0


def test_two_groups_never_share_force_or_readiness():
    s=scene((0,0,1,1)); s['hands'][0,1,:,1]=-2
    s['hand_forces'][0,0,:,1]=-10; s['hand_forces'][0,2,:,1]=-100
    v=motion_penalty(**s)
    torch.testing.assert_close(v['penalty'],torch.tensor([[-.19,0,-.19,0]]))
    s['hands'][0,1,:,1]=-1
    torch.testing.assert_close(motion_penalty(**s)['penalty'],torch.tensor([[0.,0,-.19,0]]))


def test_single_touching_hand_can_pay_but_does_not_open_gate():
    s=scene();s['hands'][...,1]=-1;s['hands'][0,0,1,1]=-2
    s['hand_forces'][0,0,0,1]=-10
    v=motion_penalty(**s)
    assert not v['ready'].any()
    assert v['penalty'][0,0].item()==pytest.approx(-.19)


def test_ready_is_instantaneous_not_a_latch():
    s=scene();s['hand_forces'][0,0,:,1]=-10
    assert motion_penalty(**s)['penalty'].sum()<0
    s['hands'][...,1]=-1
    assert motion_penalty(**s)['penalty'].sum()==0
    s['hands'][0,3,:,1]=-2
    assert motion_penalty(**s)['penalty'].sum()<0


def test_owner_permutation_and_environment_isolation():
    s=scene((0,0,1,1));s['hand_forces'][0,2,:,1]=-10
    before=motion_penalty(**s)['penalty']
    order=torch.tensor([2,0,3,1])
    for k in ('hands','hand_forces','boxes','sizes','groups','templates'):
        s[k]=s[k][:,order].repeat(2,*([1]*(s[k].ndim-1)))
    s['hand_forces'][1]=0
    after=motion_penalty(**s)['penalty']
    torch.testing.assert_close(after[0],before[0,order]);assert not after[1].any()


@pytest.mark.parametrize('bad',[{'weight':-1}, {'weight':float('nan')}, {'speed_threshold':-1},
                               {'force_epsilon':0}, {'weight':True}, {'unexpected':1}])
def test_invalid_config_rejected(bad):
    c=dict(weight=.2,speed_threshold=.05,force_epsilon=1e-6);c.update(bad)
    with pytest.raises(ValueError):validate_config(c)


def test_variant_only_adds_penalty_and_preserves_base():
    base='stage2_coupled_push_individual_goal_solo5_carry5_ground'
    variant=base+'_premature_motion'
    a=native_config(base);b=native_config(variant)
    validate_config(b['env']['stage2JointPush'].pop('premature_motion_penalty'))
    b['experiment']['name']=base
    assert a==b
    env=native_config(variant)['env']
    validate_experiment(variant,env,native_config(variant,training=True))
    validate_experiment(base,a['env'],native_config(base,training=True))
    assert checkpoint_contract(env)!=checkpoint_contract(a['env'])


def test_native_selected_side_drives_readiness_and_force_attribution():
    from utils.coupled_push_contact import group_face_geometry
    s=scene((0,0));s['hands'][...,0]=-1;s['hands'][...,1]=0
    s['hands'][0,1,:,0]=-2
    goals=torch.zeros(1,2,3);goals[...,0]=5
    members=torch.full((1,2),2);regions=torch.tensor([[0,1]])
    face=group_face_geometry(s['hands'],s['boxes'],s['sizes'],goals,s['groups'],members,regions)
    assert face['near'][0,0].all() and not face['near'][0,1].any()
    s['boxes'][...,7]=1;s['boxes'][...,8]=0
    s['hand_forces'][0,0,:,0]=-10
    v=motion_penalty(**s,near=face['near'])
    torch.testing.assert_close(v['penalty'],torch.tensor([[-.19,0.]]))
    s['hands'][0,1,:,0]=-1
    face=group_face_geometry(s['hands'],s['boxes'],s['sizes'],goals,s['groups'],members,regions)
    assert face['on_face'].all()
    assert not motion_penalty(**s,near=face['near'])['penalty'].any()
