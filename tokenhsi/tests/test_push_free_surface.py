"""Full surface distance, contact force direction, state success and bank compatibility."""
from copy import deepcopy
from isaacgym import gymapi  # before torch
import torch
import pytest
import yaml
from test_open_push import ROOT,geometry
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import push_any_surface_geometry,push_hand_contact,push_reward,OpenPushRuntime


def config():
    return yaml.safe_load((ROOT/'data/cfg/multi_agent/push_contact_at_friction06_free_surface_state09.yaml').read_text())['env']


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('sign',[-1.,1.])
@pytest.mark.parametrize('rotated',[False,True])
def test_all_faces_and_rotation_allow_arbitrary_hand_locations(axis,sign,rotated):
    e=config();roots,hands,boxes,goals,start=geometry()
    local=torch.zeros_like(hands);local[...,axis]=sign*.65
    local[..., (axis+1)%3]=torch.tensor([.3,-.2])
    if rotated:
        boxes[...,3]=torch.sin(torch.tensor(.3));boxes[...,6]=torch.cos(torch.tensor(.3))
    from utils.door_reference import rotate
    hands=boxes[:,:,None,:3]+rotate(boxes[:,:,None,3:7].expand(1,2,2,4),local)
    d,normal=push_any_surface_geometry(hands,boxes,e['pushBox']['size'])
    torch.testing.assert_close(d,torch.zeros_like(d),atol=1e-6,rtol=0)
    force=normal*2;bf=torch.zeros(1,2,3)
    assert push_hand_contact(hands,force,boxes,bf,e['pushBox']['size'],e['relationReward']['push'],goals).all()
    force[...,1,:]=0
    assert not push_hand_contact(hands,force,boxes,bf,e['pushBox']['size'],e['relationReward']['push'],goals).any()
    assert not push_hand_contact(hands,-normal*2,boxes,bf,e['pushBox']['size'],e['relationReward']['push'],goals).any()


def test_center_is_not_surface_and_corner_uses_euclidean_distance():
    e=config();roots,hands,boxes,goals,start=geometry()
    hands=boxes[:,:,None,:3].expand_as(hands).clone()
    d,_=push_any_surface_geometry(hands,boxes,e['pushBox']['size'])
    torch.testing.assert_close(d,torch.full_like(d,.65))
    hands+=torch.tensor([.75,.75,0.])
    d,normal=push_any_surface_geometry(hands,boxes,e['pushBox']['size'])
    torch.testing.assert_close(d,torch.full_like(d,2**.5*.1))
    assert torch.isfinite(normal).all()
    d2,_=push_any_surface_geometry(hands,boxes,torch.full((1,2,3),1.3))
    torch.testing.assert_close(d,d2)


def test_success_needs_raw_state_and_both_hand_contacts():
    e=config();roots,hands,boxes,goals,start=geometry()
    boxes[...,:3]=goals;hands=boxes[:,:,None,:3].expand_as(hands).clone()
    hands[...,0]=boxes[...,None,0]+.65
    flags=torch.zeros(1,2,dtype=torch.bool)
    def r(hit):return push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],hit,goal_contact=True)
    assert (r(flags)['state_score']==1).all() and not r(flags)['success'].any()
    assert r(~flags)['success'].all()
    hands[...,1,0]+=1
    assert not r(~flags)['success'].any()
    hands[...,1,0]-=1
    rt=OpenPushRuntime(1,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',e['pushBox']['size'],True)
    rt.reset(torch.arange(1),torch.ones(1,2,dtype=torch.long),boxes)
    rt.push_contact.fill_(True);rt.step(roots,hands,roots,torch.zeros(1,2),boxes,goals)
    assert rt.done.all()
    rt.push_contact.zero_();rt.step(roots,hands,roots,torch.zeros(1,2),boxes,goals)
    assert not rt.done.any()


def test_reward_option_preserves_physical_rsi_contract():
    e=config();validate_env(e)
    old=deepcopy(e);old['relationReward']['push'].pop('contact_mode')
    for row in (e,old):row['motion_file']=str(ROOT/'data/dataset_open_push_push180.yaml')
    assert bank_contract(e)==bank_contract(old)
    e['relationReward']['push']['contact_mode']='bad'
    with pytest.raises(ValueError):validate_env(e)
