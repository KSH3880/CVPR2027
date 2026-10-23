"""Frozen material face and height-free two-hand midpoint contact."""
from pathlib import Path
from isaacgym import gymapi
import torch
import yaml
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import push_face_geometry,push_hand_targets,push_hand_contact,push_reward


def scene():
    e=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint.yaml').read_text())['env']
    b=torch.zeros(2,2,13);b[...,2]=.65;b[...,6]=1
    g=b[...,:3].clone();g[...,1]+=2
    side=push_face_geometry(b,e['pushBox']['size'],g)[-1]
    return e,b,g,side


def test_bank_and_config_compatibility():
    e,_,_,_=scene();validate_env(e)
    old=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_20kg_no_tilt.yaml').read_text())['env']
    for v in (e,old):v['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
    assert bank_contract(e)==bank_contract(old)
    assert e['relationReward']!=old['relationReward']
    assert e['numEnvs']==2048


def test_face_never_reselected_by_goal_or_hands_and_follows_box_yaw():
    e,b,g,side=scene();size=e['pushBox']['size']
    initial=push_face_geometry(b,size,g,face_side=side)
    g[...,1]=-4;g[...,0]=5
    hands=torch.randn(2,2,2,3)*10
    frozen=push_face_geometry(b,size,g,hands,side)
    for i in range(5):torch.testing.assert_close(frozen[i],initial[i])
    b[...,0]+=3;b[...,5]=2**-.5;b[...,6]=2**-.5
    center,normal,_,_,actual=push_face_geometry(b,size,g,hands,side)
    torch.testing.assert_close(actual,side)
    torch.testing.assert_close(normal[...,0],torch.full((2,2),-1.))
    torch.testing.assert_close(center[...,0],torch.full((2,2),3.65))


def test_midpoint_shift_loses_contact_and_reward_but_height_is_free():
    e,b,g,side=scene();spec=e['relationReward']['push'];size=e['pushBox']['size']
    h=push_hand_targets(b,size,spec,g,face_side=side)
    _,n,t,_,_=push_face_geometry(b,size,g,face_side=side)
    forces=-n[:,:,None].expand_as(h)*2;bf=n*2
    flags=torch.zeros(2,2,dtype=torch.bool)
    def hit(h):return push_hand_contact(h,forces,b,bf,size,spec,g,side)
    def phi(h):return push_reward(h.mean(-2),h,b,g,b[...,:3],flags,e['relationReward'],size,flags,goal_contact=True,face_side=side)['push_phi']
    assert hit(h).all();torch.testing.assert_close(phi(h),torch.ones(2,2))
    spread=h.clone()
    spread[...,0,:]+=t*.3;spread[...,1,:]-=t*.3
    assert hit(spread).all();torch.testing.assert_close(phi(spread),torch.ones(2,2))
    shifted=h+t[:,:,None]*.3
    assert not hit(shifted).any();assert (phi(shifted)<.5).all()
    h[...,0,2]=.3;h[...,1,2]=1.2
    assert hit(h).all();torch.testing.assert_close(phi(h),torch.ones(2,2))
    # Averaging cannot hide a hand outside the chosen face.
    h[...,0,:2]+=t[...,:2]*1.2;h[...,1,:2]-=t[...,:2]*1.2
    assert not hit(h).any();assert (phi(h)<.5).all()


def test_goal_creation_only_updates_reset_owner_rows():
    from types import SimpleNamespace
    from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
    e,b,g,side=scene()
    task=SimpleNamespace(_relation_cfg=e['relationReward'],num_envs=2,device='cpu',cfg={'env':e},_push_only=True,
        relation_runtime=SimpleNamespace(start=b[...,:3].clone(),face_side=None),_tar_pos=torch.zeros_like(g),
        _push_goal_offsets=g-b[...,:3],_push_reset_states=b,
        _owner_push_values=lambda values,ids:values[ids])
    HumanoidMAOpenPush._reset_task(task,torch.tensor([0,1]))
    torch.testing.assert_close(task.relation_runtime.face_side,side)
    task._push_goal_offsets[0,...,1]=-2
    HumanoidMAOpenPush._reset_task(task,torch.tensor([0]))
    assert (task.relation_runtime.face_side[0]==3).all()
    torch.testing.assert_close(task.relation_runtime.face_side[1],side[1])
