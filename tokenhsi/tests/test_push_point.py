"""Center observations, fixed material point reward/contact, and AT regression."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from env.tasks.multi_agent.humanoid_ma_open_push_point import HumanoidMAOpenPushPoint,PushPointRewardRuntime
from env.tasks.multi_agent.humanoid_ma_open_push_anchor import PushPointRuntime
from utils.open_push_spec import validate_env
from utils.push_point import validate_point,validate_point_reward,local_points,world_points,point_reward_phi,point_reward_contact


def scene():
    e=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_point_goal3to5.yaml').read_text())['env']
    b=torch.zeros(2,2,13);b[...,2]=.65;b[...,6]=1
    s=torch.full((2,2),2,dtype=torch.long)
    l=local_points(s,[1.3]*3,torch.full((2,2),.6175))
    h=world_points(b,l)[:,:,None].expand(-1,-1,2,-1).clone()
    h[...,0,0]+=.2;h[...,1,0]-=.2
    return e,b,s,l,h


def test_fixed_height_is_identifiable_without_extra_observation():
    e,*_=scene();validate_env(e);validate_point(e);validate_point_reward(e)
    assert e['pushContactPoint']['height_range_m']==[.6175,.6175]
    bad=deepcopy(e);bad['pushContactPoint']['height_range_m']=[.585,.65]
    with pytest.raises(ValueError,match='fixed point height'):validate_point_reward(bad)
    bad=deepcopy(e);bad['pushPointReward']['z_tolerance_m']=.05
    with pytest.raises(ValueError):validate_point_reward(bad)


def test_point_reward_and_contact_reject_wrong_midpoint_but_allow_hand_spacing():
    e,b,s,l,h=scene();r=e['pushPointReward'];p=e['relationReward']['push']
    f=torch.zeros_like(h);f[...,1]=-2
    bf=torch.zeros_like(b[...,:3]);bf[...,1]=4
    phi=lambda x:point_reward_phi(x,b,[1.3]*3,s,l,r)
    hit=lambda x:point_reward_contact(x,f,b,bf,[1.3]*3,p,s,l,r)
    torch.testing.assert_close(phi(h),torch.ones(2,2));assert hit(h).all()
    high=h.clone();high[...,2]+=.08
    assert (phi(high)<.02).all();assert not hit(high).any()
    lateral=h.clone();lateral[...,0]+=.3
    assert (phi(lateral)<.5).all();assert not hit(lateral).any()
    symmetric=h.clone();symmetric[...,0,2]+=.2;symmetric[...,1,2]-=.2
    torch.testing.assert_close(phi(symmetric),torch.ones(2,2));assert hit(symmetric).all()
    far=h.clone();far[...,0,1]-=.5;far[...,1,1]+=.5
    assert not hit(far).any()


def test_point_reward_follows_box_rotation_and_translation():
    e,b,s,l,h=scene();old=h.clone()
    b[...,0]+=2.;b[...,5]=2**-.5;b[...,6]=2**-.5
    moved=world_points(b,l)[:,:,None].expand(-1,-1,2,-1).clone()
    moved[...,0,1]+=.2;moved[...,1,1]-=.2
    torch.testing.assert_close(point_reward_phi(moved,b,[1.3]*3,s,l,e['pushPointReward']),torch.ones(2,2))
    assert (point_reward_phi(old,b,[1.3]*3,s,l,e['pushPointReward'])<.01).all()


@pytest.mark.parametrize('band',[False,True])
def test_point_runtime_changes_push_only_and_preserves_at_saturation(band):
    e,b,s,l,h=scene();g=b[...,:3].clone();g[...,1]+=4
    if band:e['pushPointReward']=dict(height_band_m=[.585,.65],z_reward_sigma_m=.04)
    args=(2,e['relationGraph'],e['relationReward'],1/30,'cpu',[1.3]*3,True)
    face=PushPointRuntime(*args,point_spec=e['pushContactPoint'])
    point=PushPointRewardRuntime(*args,point_spec=e['pushPointReward'])
    for runtime in (face,point):
        runtime.reset(torch.arange(2),torch.ones(2,2,dtype=torch.long),b)
        runtime.face_side=s;runtime.point_local[:]=l
    high=h.clone();high[...,2]+=.08;roots=h.mean(-2)
    a=face.step(roots,high,None,None,b,g);c=point.step(roots,high,None,None,b,g)
    assert (c['state']<a['state']).all()
    for key in ('progress','bonus','success'):torch.testing.assert_close(a[key],c[key])
    torch.testing.assert_close(face.phi[:,1::2],point.phi[:,1::2])
    for runtime in (face,point):runtime.push_contact[:]=True
    a=face.step(roots,high,None,None,b,g);c=point.step(roots,high,None,None,b,g)
    torch.testing.assert_close(a['reward'],c['reward'])
    a=face.step(roots,high,None,None,b,b[...,:3]);c=point.step(roots,high,None,None,b,b[...,:3])
    assert c['success'].all();torch.testing.assert_close(a['reward'],c['reward'])
