"""Anchor observations, face contact without midpoint targets, and AT isolation."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from utils.push_point import local_points,world_points,face_contact,face_phi,validate_point,anchor_view
from env.tasks.multi_agent.humanoid_ma_open_push_anchor import HumanoidMAOpenPushAnchor,PushPointRuntime
from env.tasks.multi_agent.open_push_reward import PushOnlyRuntime
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract,OpenPushRSIBank


def scene():
    env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/push_fixed130_anchor_goal3to5.yaml').read_text())['env']
    b=torch.zeros(2,2,13);b[...,2]=.65;b[...,6]=1
    side=torch.full((2,2),2,dtype=torch.long)
    local=local_points(side,[1.3]*3,torch.full((2,2),.6))
    h=world_points(b,local)[:,:,None].expand(-1,-1,2,-1).clone()
    h[...,0,0]+=.2;h[...,1,0]-=.2
    return env,b,side,local,h


def test_fixed_anchor_config_and_distinct_rsi_contract():
    e,*_=scene();validate_env(e);validate_point(e)
    assert e['pushBox']=={'size':[1.3]*3,'mass':20.,'friction':.6}
    assert e['pushGoalRandomization']['distance_range']==[3.,5.]
    assert e['pushContactPoint']['height_range_m']==[.585,.65]
    assert HumanoidMAOpenPushAnchor.get_clean_object_obs_size(None)==30
    old=deepcopy(e);old.pop('pushContactPoint')
    for v in (old,e):v['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
    assert bank_contract(old)!=bank_contract(e)
    bad=deepcopy(e);bad['pushContactPoint']['height_range_m']=[.65,.585]
    with pytest.raises(ValueError):validate_point(bad)


def test_material_point_follows_full_rotation_and_translation():
    _,b,s,l,_=scene()
    torch.testing.assert_close(world_points(b,l)[...,2],torch.full((2,2),.6))
    b[...,0]=3.;b[...,3]=2**-.5;b[...,6]=2**-.5
    p=world_points(b,l)
    torch.testing.assert_close(p[...,0],torch.full((2,2),3.))
    torch.testing.assert_close(p[...,1],torch.full((2,2),.05),atol=1e-6,rtol=0)
    torch.testing.assert_close(p[...,2],torch.zeros(2,2),atol=1e-6,rtol=0)


def test_face_contact_has_no_midpoint_xy_or_height_gate():
    e,b,s,l,h=scene();push=e['relationReward']['push']
    f=torch.zeros_like(h);f[...,1]=-2
    bf=torch.zeros_like(b[...,:3]);bf[...,1]=4
    hit=lambda hands:face_contact(hands,f,b,bf,[1.3]*3,push,s,l)
    assert hit(h).all()
    # Both hands move away from the anchor, staying on the material face.
    shifted=h.clone();shifted[...,0]+=.3;shifted[...,2]+=.3
    assert hit(shifted).all()
    torch.testing.assert_close(face_phi(shifted,b,[1.3]*3,s,l),torch.ones(2,2))
    # Changing the observed anchor height cannot change face reward/contact.
    changed=l.clone();changed[...,2]-=.05
    torch.testing.assert_close(face_phi(shifted,b,[1.3]*3,s,changed),face_phi(shifted,b,[1.3]*3,s,l))
    assert face_contact(shifted,f,b,bf,[1.3]*3,push,s,changed).all()
    far=h.clone();far[...,0,1]-=.4;far[...,1,1]+=.4
    assert not hit(far).any()
    assert not face_contact(h,torch.zeros_like(f),b,bf,[1.3]*3,push,s,l).any()
    above=h.clone();above[...,2]=1.6
    assert not hit(above).any()


def test_reward_preserves_at_progress_and_saturation():
    e,b,s,l,h=scene();g=b[...,:3].clone();g[...,1]+=4
    args=(2,e['relationGraph'],e['relationReward'],1/30,'cpu',[1.3]*3,True)
    old=PushOnlyRuntime(*args);new=PushPointRuntime(*args,point_spec=e['pushContactPoint'])
    ids=torch.arange(2);tasks=torch.ones(2,2,dtype=torch.long)
    for r in (old,new):r.reset(ids,tasks,b);r.face_side=s
    new.point_local[:]=l;roots=h.mean(-2)
    a=old.step(roots,h,None,None,b,g);c=new.step(roots,h,None,None,b,g)
    torch.testing.assert_close(a['reward'],c['reward'])
    high=h.clone();high[...,2]+=.1
    a=old.step(roots,high,None,None,b,g);c=new.step(roots,high,None,None,b,g)
    torch.testing.assert_close(c['state'],a['state'])
    torch.testing.assert_close(c['state'],new.step(roots,h,None,None,b,g)['state'])
    torch.testing.assert_close(a['progress'],c['progress']);torch.testing.assert_close(a['success'],c['success'])
    for r in (old,new):r.push_contact[:]=True
    a=old.step(roots,high,None,None,b,g);c=new.step(roots,high,None,None,b,g)
    torch.testing.assert_close(a['reward'],c['reward'])
    a=old.step(roots,high,None,None,b,b[...,:3]);c=new.step(roots,high,None,None,b,b[...,:3])
    assert c['success'].all();torch.testing.assert_close(a['reward'],c['reward'])


def test_anchor_view_replaces_pose_and_rigid_velocity_only():
    _,b,s,l,_=scene();m=2;h=3;o=30;g=1
    end=m*h+m*o+m*g
    obs=torch.randn(2,end+6*7+20)
    objects=obs[:,m*h:m*h+2*o].reshape(2,2,o)
    poses=obs[:,end:end+42].reshape(2,6,7)
    poses[:,:,3:7]=torch.tensor([0.,0.,0.,1.])
    poses[:,2:4,:3]=b[...,:3]
    old=obs.clone();v=objects[...,:3].clone();omega=objects[...,3:6].clone()
    result=anchor_view(obs,m,[h,o,g],l)
    torch.testing.assert_close(obs,old,rtol=0,atol=0)
    new_o=result[:,m*h:m*h+2*o].reshape(2,2,o)
    new_p=result[:,end:end+42].reshape(2,6,7)
    torch.testing.assert_close(new_p[:,2:4,:3],world_points(b,l))
    torch.testing.assert_close(new_o[...,:3],v+torch.cross(omega,l,dim=-1))
    torch.testing.assert_close(new_o[...,3:],objects[...,3:],rtol=0,atol=0)
    for idx in (0,1,4,5):torch.testing.assert_close(new_p[:,idx],poses[:,idx],rtol=0,atol=0)
    torch.testing.assert_close(result[:,-20:],obs[:,-20:],rtol=0,atol=0)


def test_partial_reset_preserves_other_owner_points():
    e,b,s,l,h=scene()
    task=SimpleNamespace(num_envs=2,device='cpu',cfg={'env':e},_relation_cfg=e['relationReward'],_push_only=True,
        relation_runtime=SimpleNamespace(start=b[...,:3].clone(),face_side=None,point_local=torch.zeros_like(l)),
        _tar_pos=torch.zeros_like(l),_push_goal_offsets=torch.tensor([0.,4.,0.]).expand_as(l).clone(),
        _push_reset_states=b,_point_heights_physical=torch.tensor([[.585,.65],[.6,.62]]),
        _owner_push_values=lambda values,ids:values[ids],_owner_push_sizes=lambda ids:torch.full((len(ids),2,3),1.3))
    instance=object.__new__(HumanoidMAOpenPushAnchor)
    instance.__dict__.update(task.__dict__);task=instance
    HumanoidMAOpenPushAnchor._reset_task(task,torch.tensor([0,1]))
    saved=task.relation_runtime.point_local[1].clone();before=task.relation_runtime.point_local[0].clone()
    task._point_heights_physical[0]+=.01
    HumanoidMAOpenPushAnchor._reset_task(task,torch.tensor([0]))
    torch.testing.assert_close(task.relation_runtime.point_local[1],saved,rtol=0,atol=0)
    assert not torch.equal(task.relation_runtime.point_local[0],before)


def test_screened_rsi_subset_is_opt_in_and_keeps_amp_weights():
    e,*_=scene()
    library=SimpleNamespace(gts=torch.zeros(1),num_motions=lambda:3,
        _motion_lengths=torch.ones(3),_motion_weights=torch.tensor([1.,2.,3.]))
    proposals={'push':[dict(motion_id=0,time=.5,phase=.5,box_progress=.5)]}
    weights=library._motion_weights.clone()
    with pytest.raises(ValueError,match='No PD-valid'):
        OpenPushRSIBank(e['openPushRSI'],{'push':library},{},proposals,active_skills=('push',))
    bank=OpenPushRSIBank(e['openPushRSI'],{'push':library},{},proposals,active_skills=('push',),allow_missing_clips=True)
    assert (bank.sample('push',128)['motion_id']==0).all()
    torch.testing.assert_close(library._motion_weights,weights,rtol=0,atol=0)
    with pytest.raises(ValueError,match='No positive'):
        OpenPushRSIBank(e['openPushRSI'],{'push':library},{},{'push':[]},active_skills=('push',),allow_missing_clips=True)
