"""Box-centred goal bounds, joint clearance, RSI invariance and explicit transfer."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from isaacgym import gymapi  # before torch
import pytest
import torch
import yaml
from utils.push_scene import validate_scene, sample_random_at_scene, random_at_goal_valid
from utils.open_push_rsi import bank_contract
from env.tasks.multi_agent.open_push_reward import push_rsi_hand_targets
from utils.relation_task_spec import checkpoint_metadata

ROOT = Path(__file__).resolve().parents[2]
BASE = 'push_contact_face_at_tilt_soft'


def config(suffix='_3to5m'):
    env=yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{BASE}{suffix}.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_open_push.yaml'
    return env


@pytest.mark.parametrize('seed',[0,42,356])
def test_full_training_batch_radial_bounds_and_clearance(seed):
    env=config();validate_scene(env)
    xy,q,goals=sample_random_at_scene(2048,env,'cpu',torch.Generator().manual_seed(seed))
    distance=(goals-xy).norm(dim=-1)
    assert ((distance>=3-1e-6)&(distance<=5+1e-6)).all()
    assert random_at_goal_valid(xy,q,goals,env).all()
    assert distance.min()<3.02 and distance.max()>4.98
    for agent in range(2):
        delta=goals[:,agent]-xy[:,agent]
        quadrant=(delta[:,0]>0).long()+2*(delta[:,1]>0).long()
        assert (torch.bincount(quadrant,minlength=4)>150).all()
    bad=goals.clone();bad[:,0]=xy[:,0]+(goals[:,0]-xy[:,0])/distance[:,0,None]*5.01
    assert not random_at_goal_valid(xy,q,bad,env).any()
    bad[:,0]=xy[:,0]+(goals[:,0]-xy[:,0])/distance[:,0,None]*2.99
    assert not random_at_goal_valid(xy,q,bad,env).any()


@pytest.mark.parametrize('change',[{'max_distance':3},{'min_distance':True},
    {'max_distance':float('nan')},{'sampling':'unknown'},{'reach':2}])
def test_reject_invalid_radial_spec(change):
    env=config();env['pushScene']['random_at_goal'].update(change)
    with pytest.raises(ValueError):validate_scene(env)


def test_only_scene_changes_and_bank_reusable():
    old,new=config(''),config()
    assert bank_contract(old)==bank_contract(new)
    comparable=deepcopy(new);comparable['pushScene']=old['pushScene']
    assert comparable==old


@pytest.mark.parametrize('angle',[0,.7,math.pi,5.1])
def test_rsi_contact_targets_invariant_to_goal_distance(angle):
    env=config();xy,q,_=sample_random_at_scene(12,env,'cpu',torch.Generator().manual_seed(9))
    boxes=torch.zeros(12,2,13);boxes[...,:2]=xy;boxes[...,2]=.65;boxes[...,3:7]=q
    direction=torch.tensor([math.cos(angle),math.sin(angle)])
    def targets(distance):
        goals=boxes[...,:3].clone();goals[...,:2]+=distance*direction
        return push_rsi_hand_targets(boxes,goals,env['pushBox']['size'],env['relationReward'])
    torch.testing.assert_close(targets(3.),targets(5.),atol=1e-5,rtol=1e-5)


def transfer_fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('transfer_goals',ROOT/'tokenhsi/scripts/multi_agent/transfer_push_goal_distance_checkpoint.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old,new=config(''),config()
    contract=dict(push_scene=dict(revision=1,parameters=new['pushScene'],box_reset=new['box']['reset']),
        push_at_reward=dict(parameters=new['relationReward']),
        rsi_bank_sha256=hashlib.sha256(Path(new['openPushRSI']['bank_file']).read_bytes()).hexdigest())
    audit=dict(device='cpu',reset_agent_samples=1000,both_assignments_rsi=True,bank_preserved=True,contract=contract)
    audit_path=tmp_path/'audit.json';audit_path.write_text(json.dumps(audit))
    source_contract=deepcopy(contract);source_contract['push_scene']['parameters']=old['pushScene']
    train=yaml.safe_load((ROOT/f'tokenhsi/data/cfg/train/rlg/{BASE}.yaml').read_text())
    weights=dict(relation_metadata=checkpoint_metadata(old['relationReward']),open_push_contract=source_contract,
        relation_task_instance=dict(graph=old['relationGraph'],num_agents=2,num_objects=2),
        relation_experiment_config=dict(env=old,train=train),model={'weight':torch.ones(2)},
        optimizer=dict(state={0:dict(step=7)},param_groups=[]),epoch=12,frame=345,
        env_state={'old':1},last_mean_rewards=1.)
    return module,audit_path,weights


def test_transfer_preserves_weights_and_rejects_unrelated_changes(tmp_path):
    module,audit,weights=transfer_fixture(tmp_path)
    source=tmp_path/'source.pth';out=tmp_path/'target.pth';torch.save(weights,source)
    module.transfer(source,out,audit)
    result=torch.load(out,map_location='cpu')
    torch.testing.assert_close(result['model']['weight'],weights['model']['weight'])
    assert (result['epoch'],result['frame'])==(12,345)
    assert result['optimizer']['state']=={} and result['env_state'] is None
    with pytest.raises(ValueError,match='already exists'):module.transfer(source,out,audit)
    weights['open_push_contract']['push_scene']['box_reset']['ownerLocoDistanceRange']=[1.,3.]
    torch.save(weights,source)
    with pytest.raises(ValueError,match='Source physics'):module.transfer(source,tmp_path/'bad.pth',audit)
