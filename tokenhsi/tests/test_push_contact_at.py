"""Independent contact/placement rewards and legacy-mode regression."""
from copy import deepcopy
from isaacgym import gymapi  # before torch
import torch
import pytest
from test_open_push import env, geometry, ROOT
import yaml
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import push_reward, OpenPushRuntime


def config():
    return yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_push180_contact_at.yaml').read_text())['env']


def test_config_and_rsi_screen_preserved():
    e=config();validate_env(e)
    legacy=deepcopy(e);legacy['relationReward'].pop('contact_at')
    for row in (e,legacy):row['motion_file']=str(ROOT/'data/dataset_open_push_push180.yaml')
    assert bank_contract(e)==bank_contract(legacy)
    e['relationReward']['contact_at']['satisfaction_threshold']=1.1
    with pytest.raises(ValueError):validate_env(e)


def test_at_progress_01_is_dense_near_goal_and_push_keeps_05():
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01.yaml').read_text())['env']
    validate_env(e)
    old=yaml.safe_load((ROOT/'data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25.yaml').read_text())['env']
    for row in (e,old):row['motion_file']=str(ROOT/'data/dataset_open_push_push180.yaml')
    assert bank_contract(e)==bank_contract(old)
    roots,hands,boxes,goals,start=geometry()
    flags=torch.zeros(1,2,dtype=torch.bool)
    roots[...,:2]=boxes[...,:2]
    def reward(distance):
        goals.copy_(boxes[...,:3]);goals[...,0]+=distance;goals[...,2]+=.01
        return push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],flags)
    assert (reward(.3)['progress']<reward(.15)['progress']).all()
    torch.testing.assert_close(reward(.1)['progress'],reward(.05)['progress'])
    torch.testing.assert_close(reward(.3)['progress'],torch.full_like(flags,.2*(1+1/1.2),dtype=torch.float))
    near=reward(.3)['progress'];roots[...,0]+=.4
    torch.testing.assert_close(reward(.3)['progress'],near)
    roots[...,0]+=.2
    assert (reward(.3)['progress']<near).all()


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),True])
def test_invalid_at_progress_delta_rejected(value):
    e=config();e['relationReward']['contact_at']['at_progress_delta']=value
    with pytest.raises(ValueError):validate_env(e)


def test_at_rewards_body_movement_without_hand_history():
    e=config();roots,hands,boxes,goals,start=geometry()
    flags=torch.zeros(1,2,dtype=torch.bool)
    def reward():return push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],flags)
    before=reward();boxes[...,1]=.75;middle=reward()
    assert (middle['phi']>before['phi']).all() and (middle['progress']>before['progress']).all()
    boxes[...,:3]=goals;boxes[...,7:13]=5
    assert reward()['success'].all()  # AT has no velocity or contact-credit condition.
    boxes[...,2]+=.01
    assert not reward()['success'].any()
    boxes[...,2]-=.01;boxes[...,1]-=.5
    assert not reward()['success'].any()


def test_contact_reward_is_independent_of_at_and_requires_both_hands_near():
    e=config();roots,hands,boxes,goals,start=geometry()
    flags=torch.zeros(1,2,dtype=torch.bool)
    def reward(contact=flags):return push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],contact)
    near=reward();hands[...,0,0]+=2
    far=reward()
    assert (far['state']<near['state']).all()
    torch.testing.assert_close(far['phi'],near['phi'])
    touch=reward(torch.ones_like(flags))
    assert touch['push_success'].all() and not touch['success'].any()
    assert (touch['bonus']==.2).all()


def test_runtime_keeps_edge_success_separate_and_allows_release_at_goal():
    e=config();roots,hands,boxes,goals,start=geometry(2)
    rt=OpenPushRuntime(2,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',e['pushBox']['size'])
    ids=torch.arange(2);rt.reset(ids,torch.ones(2,2,dtype=torch.long),boxes)
    rt.push_contact.fill_(True)
    def step():return rt.step(roots,hands,roots,torch.zeros(2,2),boxes,goals)
    step()
    assert rt.own_success[:,::2].all() and not rt.own_success[:,1::2].any() and not rt.done.any()
    rt.push_contact.zero_();boxes[...,:3]=goals;hands+=10
    out=step()
    assert not rt.own_success[:,::2].any() and rt.own_success[:,1::2].all() and rt.done.all()
    torch.testing.assert_close(out['reward'],torch.full((2,2),1.2))
    assert not rt.push_credit.any()
    rt.reset(ids[:1],torch.zeros(1,2,dtype=torch.long),boxes[:1])
    assert not rt.done[0].any() and rt.done[1].all()



def test_state_only_success_uses_raw_mean_before_saturation():
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/push_contact_at_friction06_fixed_size_state09.yaml').read_text())['env']
    validate_env(e)
    assert 'pushSizeRandomization' not in e
    assert e['pushBox']=={'size':[1.3]*3,'mass':20.,'friction':.6}
    assert e['relationGraph']['task_probabilities']=={'OPEN':0.,'PUSH':1.}
    roots,hands,boxes,goals,start=geometry()
    from env.tasks.multi_agent.open_push_reward import push_hand_targets
    flags=torch.zeros(1,2,dtype=torch.bool)
    def reward(contact=flags):return push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],contact,goal_contact=True)
    boxes[...,:3]=goals;boxes[...,2]+=.02
    hands.copy_(push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals))
    r=reward();assert r['success'].all() and (r['state_score']>=.9).all()
    legacy=deepcopy(e['relationReward']);legacy['contact_at'].pop('success_mode')
    assert not push_reward(roots,hands,boxes,goals,start,flags,legacy,e['pushBox']['size'],flags,goal_contact=True)['success'].any()
    hands+=10
    assert not reward(torch.ones_like(flags))['success'].any()  # contact cannot inflate raw state
    hands.copy_(push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals))
    boxes[...,0]+=.5
    assert not reward()['success'].any()
    e['relationReward']['contact_at']['success_mode']='unknown'
    with pytest.raises(ValueError):validate_env(e)


def test_state_threshold_is_inclusive_and_runtime_matches():
    e=yaml.safe_load((ROOT/'data/cfg/multi_agent/push_contact_at_friction06_fixed_size_state09.yaml').read_text())['env']
    roots,hands,boxes,goals,start=geometry()
    from env.tasks.multi_agent.open_push_reward import push_hand_targets
    boxes[...,:3]=goals
    hands.copy_(push_hand_targets(boxes,e['pushBox']['size'],e['relationReward']['push'],goals));hands[...,1]+=.1
    flags=torch.zeros(1,2,dtype=torch.bool)
    def r():return push_reward(roots,hands,boxes,goals,start,flags,e['relationReward'],e['pushBox']['size'],flags,goal_contact=True)
    threshold=r()['state_score'][0,0].item()
    e['relationReward']['contact_at']['satisfaction_threshold']=threshold
    assert r()['success'].all()
    e['relationReward']['contact_at']['satisfaction_threshold']=threshold+1e-5
    assert not r()['success'].any()
    e['relationReward']['contact_at']['satisfaction_threshold']=.9
    rt=OpenPushRuntime(1,e['relationGraph'],e['relationReward'],e['doorHold'],1/30,'cpu',e['pushBox']['size'],goal_contact=True)
    rt.reset(torch.arange(1),torch.ones(1,2,dtype=torch.long),boxes)
    rt.step(roots,hands,roots,torch.zeros(1,2),boxes,goals)
    assert rt.done.all()
    hands+=10;rt.step(roots,hands,roots,torch.zeros(1,2),boxes,goals)
    assert not rt.done.any()
