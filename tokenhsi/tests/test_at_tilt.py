"""AT tilt gate boundaries, reward independence and strict migration."""
from copy import deepcopy
import math
from pathlib import Path
from isaacgym import gymapi
import pytest
import torch
import yaml
from test_open_push import geometry
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from utils.at_tilt_checkpoint import BASE,CFG,migrate_checkpoint
from env.tasks.multi_agent.open_push_reward import contact_at_reward


def env():return yaml.safe_load((CFG/(BASE+'_tilt035.yaml')).read_text())['env']


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('theta',[0.,.3499,.3501,math.pi/2,math.pi])
def test_tilt_gate_and_yaw(axis,theta):
    e=env();roots,hands,boxes,goals,_=geometry()
    boxes[...,:3]=goals;boxes[...,3:7]=0
    boxes[...,3+axis]=math.sin(theta/2);boxes[...,6]=math.cos(theta/2)
    flags=torch.zeros(1,2,dtype=torch.bool)
    old=deepcopy(e['relationReward']);old['contact_at'].pop('max_tilt')
    def calc(c):return contact_at_reward(roots,hands,boxes,goals,c,e['pushBox']['size'],flags,True)
    baseline=calc(old);new=calc(e['relationReward'])
    assert baseline['success'].all()
    assert bool(new['success'].all())==(axis==2 or theta<=.35)
    for key in ('phi','state_score','error','push_phi','push_success','grounded'):
        torch.testing.assert_close(new[key],baseline[key])
    if axis==2 or theta<=.35:
        for key in new:torch.testing.assert_close(new[key],baseline[key])
    else:
        assert not new['bonus'].any()
        assert (new['state']<baseline['state']).all() and (new['progress']<=baseline['progress']).all()


def test_unsuccessful_dense_reward_and_contact_unchanged():
    e=env();roots,hands,boxes,goals,_=geometry()
    boxes[...,3]=math.sin(.8/2);boxes[...,6]=math.cos(.8/2)
    old=deepcopy(e['relationReward']);old['contact_at'].pop('max_tilt')
    for contact in (torch.zeros(1,2,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool)):
        a=contact_at_reward(roots,hands,boxes,goals,old,e['pushBox']['size'],contact,True)
        b=contact_at_reward(roots,hands,boxes,goals,e['relationReward'],e['pushBox']['size'],contact,True)
        assert not a['success'].any()
        for key in a:torch.testing.assert_close(a[key],b[key])


def test_only_tilt_and_same_rsi_contract():
    target=env();validate_env(target)
    source=yaml.safe_load((CFG/(BASE+'.yaml')).read_text())['env']
    stripped=deepcopy(target);stripped['relationReward']['contact_at'].pop('max_tilt')
    assert stripped==source
    for row in (target,source):row['motion_file']='tokenhsi/data/dataset_open_push_push180.yaml'
    assert bank_contract(target)==bank_contract(source)


@pytest.mark.parametrize('value',[0,-1,True,float('nan'),float('inf'),math.pi/2,math.pi])
def test_invalid_limit(value):
    e=env();e['relationReward']['contact_at']['max_tilt']=value
    with pytest.raises(ValueError):validate_env(e)


def test_boundary_inclusive():
    e=env();roots,hands,boxes,goals,_=geometry();boxes[...,:3]=goals
    boxes[...,3]=math.sin(.35/2);boxes[...,6]=math.cos(.35/2)
    r=contact_at_reward(roots,hands,boxes,goals,e['relationReward'],e['pushBox']['size'],None,True)
    assert r['success'].all()


def migration_fixture():
    import hashlib
    from utils.relation_task_spec import checkpoint_metadata
    source=yaml.safe_load((CFG/(BASE+'.yaml')).read_text())['env']
    source['motion_file']='tokenhsi/data/dataset_open_push_push180.yaml'
    train=yaml.safe_load(Path('tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml').read_text())
    return dict(relation_experiment_config={'env':source,'train':train},relation_metadata=checkpoint_metadata(source['relationReward']),amp_enabled=True,
        open_push_contract=dict(contact_at=deepcopy(source['relationReward']['contact_at']),object_obs_size=40,
            rsi_bank_sha256=hashlib.sha256(Path(source['openPushRSI']['bank_file']).read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(Path(source['motion_file']).read_bytes()).hexdigest()),
        epoch=6000,frame=6000*131072,model={'x':torch.randn(4)},optimizer={'state':torch.randn(4)},running_mean_std={'x':torch.randn(4)})


def test_migration_preserves_all_learned_state_and_source():
    w=migration_fixture();before=deepcopy(w['relation_metadata']);c=migrate_checkpoint(w)
    for key in ('model','optimizer','running_mean_std'):assert c[key] is w[key]
    assert c['epoch']==w['epoch'] and c['frame']==w['frame']
    assert w['relation_metadata']==before and 'max_tilt' not in w['open_push_contract']['contact_at']
    assert c['open_push_contract']['contact_at']['max_tilt']==.35


@pytest.mark.parametrize('kind',['reward','input','amp','bank','network'])
def test_migration_refuses_other_changes(kind):
    w=migration_fixture()
    if kind=='reward':w['relation_experiment_config']['env']['relationReward']['contact_at']['near_distance_scale']=5
    if kind=='input':w['open_push_contract']['object_obs_size']=30
    if kind=='amp':w['amp_enabled']=False
    if kind=='bank':w['open_push_contract']['rsi_bank_sha256']='bad'
    if kind=='network':w['relation_experiment_config']['train']['params']['network']['transformer']['num_layers']=1
    with pytest.raises(ValueError):migrate_checkpoint(w)
