"""Independent PUSH/AT edges and paired placement reward regressions."""
from pathlib import Path
from copy import deepcopy
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime,push_hand_targets
from env.tasks.multi_agent.edge_context_reward import goal_success

ROOT=Path(__file__).resolve().parents[2]


def scene():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_only_random_scene_at.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    boxes=torch.zeros(2,2,13);boxes[...,2]=.65;boxes[...,6]=1
    goals=boxes[...,:3].clone();goals[...,1]+=2
    roots=boxes[...,:3].clone();roots[...,1]-=1
    rt=OpenPushRuntime(2,env['relationGraph'],env['relationReward'],env['doorHold'],1/30,'cpu',env['pushBox']['size'],env['manipulationContact'])
    rt.reset(torch.arange(2),torch.ones(2,2,dtype=torch.long),boxes)
    rt.initialize_manipulation(torch.arange(2),boxes,goals,torch.zeros(2,2))
    hands=push_hand_targets(boxes,env['pushBox']['size'],env['relationReward']['push'])
    return env,rt,roots,hands,boxes,goals


def step(rt,roots,hands,boxes,goals):
    return rt.step(roots,hands,torch.zeros_like(roots),torch.zeros_like(rt.elapsed),boxes,goals)


def test_config_bank_contract_and_checkpoint_separation():
    import json
    env,rt,*_=scene();validate_env(env)
    assert json.loads(Path(env['openPushRSI']['bank_file']).read_text())['contract']==bank_contract(env)
    old=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/push_toy_only_random_scene.yaml').read_text())['env']
    assert old['relationReward']!=env['relationReward']
    assert old['pushBox']==env['pushBox'] and old['pushScene']==env['pushScene']
    bad=deepcopy(env);bad['relationGraph'].pop('push_only');bad['numObjects']=4
    with pytest.raises(ValueError):validate_env(bad)


def test_hand_success_is_independent_of_at_and_contact_credit():
    env,rt,roots,hands,boxes,goals=scene()
    result=step(rt,roots,hands,boxes,goals)
    assert rt.own_success[:,::2].all() and not rt.own_success[:,1::2].any()
    assert not result['success'].any() and not rt.push_hand_distance.any()
    torch.testing.assert_close(result['edge_state'][:,::2],torch.full((2,2),.2))
    torch.testing.assert_close(result['edge_bonus'][:,::2],torch.full((2,2),.2))


def test_at_shaping_improves_without_hands_or_force():
    env,rt,roots,hands,boxes,goals=scene();hands+=10
    before=step(rt,roots,hands,boxes,goals)
    boxes[...,1]+=1
    after=step(rt,roots,hands,boxes,goals)
    assert (after['edge_state'][:,1::2]>before['edge_state'][:,1::2]).all()
    assert (after['edge_progress'][:,1::2]>before['edge_progress'][:,1::2]).all()
    assert not rt.clean_push_contact.any() and not rt.push_hand_distance.any()


def test_at_completion_saturates_paired_push_and_releases_when_goal_lost():
    env,rt,roots,hands,boxes,goals=scene();hands+=10;boxes[...,:3]=goals
    result=step(rt,roots,hands,boxes,goals)
    assert result['success'].all() and not rt.own_success[:,::2].any()
    assert rt.own_success[:,1::2].all()
    torch.testing.assert_close(goal_success(rt.own_success,rt.graph),rt.done)
    torch.testing.assert_close(result['reward'],torch.full((2,2),1.14))
    boxes[...,1]-=.3
    assert not step(rt,roots,hands,boxes,goals)['success'].any()
    assert rt.achieved[:,1::2].all()


@pytest.mark.parametrize('offset,success',[(.10,True),(.11,False)])
def test_at_uses_original_phi_threshold_and_strict_z(offset,success):
    env,rt,roots,hands,boxes,goals=scene();boxes[...,:3]=goals;boxes[...,0]+=offset
    assert bool(step(rt,roots,hands,boxes,goals)['success'].all())==success
    boxes[...,0]-=offset;boxes[...,2]+=.002
    assert not step(rt,roots,hands,boxes,goals)['success'].any()


def test_owner_sum_teammate_mean_and_partial_reset():
    env,rt,roots,hands,boxes,goals=scene();hands[:,1]+=10;boxes[:,0,:3]=goals[:,0]
    result=step(rt,roots,hands,boxes,goals)
    local=result['edge_state']+result['edge_progress']+result['edge_bonus']
    local=local.reshape(2,2,2).sum(-1)
    torch.testing.assert_close(result['reward'],.9*local+.1*local.flip(-1)/2)
    before=rt.achieved[1].clone()
    rt.reset(torch.tensor([0]),torch.ones(1,2,dtype=torch.long),boxes[:1])
    assert not rt.achieved[0].any();torch.testing.assert_close(rt.achieved[1],before)
    assert not rt.graph.required_goal[:,::2].any() and rt.graph.required_goal[:,1::2].all()


def test_split_runtime_never_calls_combined_or_open_reward(monkeypatch):
    import env.tasks.multi_agent.open_push_reward as module
    env,rt,roots,hands,boxes,goals=scene()
    def forbidden(*args,**kwargs):raise AssertionError('combined reward called')
    monkeypatch.setattr(module,'push_reward',forbidden);monkeypatch.setattr(module,'door_reward',forbidden)
    assert torch.isfinite(step(rt,roots,hands,boxes,goals)['reward']).all()
