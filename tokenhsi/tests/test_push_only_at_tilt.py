"""Physical PUSH-only 570D experiment preserves everything except the AT gate."""
from copy import deepcopy
from pathlib import Path
from isaacgym import gymapi
import pytest
import torch
import yaml
from utils.at_tilt_checkpoint import CFG,PUSH_BASE,migrate_push_only_checkpoint
from utils.push_only import training_contract
from utils.relation_task_spec import checkpoint_metadata
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract
from test_open_push import geometry
from env.tasks.multi_agent.open_push_reward import contact_at_reward


def configs():
    source=yaml.safe_load((CFG/(PUSH_BASE+'.yaml')).read_text())['env']
    target=yaml.safe_load((CFG/(PUSH_BASE+'_tilt035.yaml')).read_text())['env']
    return source,target


def test_same_input_amp_rsi_and_all_other_settings():
    a,b=configs();validate_env(b)
    c=deepcopy(b);assert c['relationReward']['contact_at'].pop('max_tilt')==.35
    assert a==c and b['pushOnly']
    for row in (a,b):row['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
    assert bank_contract(a)==bank_contract(b)
    ca=training_contract(a,'tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml')
    cb=training_contract(b,'tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml')
    assert cb['object_obs_size']==30 and cb['policy_observation']=='push_no_doors_30d_v2'
    cb['reward']['contact_at'].pop('max_tilt');assert ca==cb


def checkpoint():
    a,b=configs();a['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
    train_path='tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml'
    return dict(relation_experiment_config=dict(env=a,train=yaml.safe_load(Path(train_path).read_text()),observation_size=570),
        open_push_contract=training_contract(a,train_path),relation_metadata=checkpoint_metadata(a['relationReward']),amp_enabled=True,
        model={'actor':torch.randn(4)},optimizer={'state':torch.randn(4)},running_mean_std={'x':torch.randn(4)},amp_input_mean_std={'x':torch.randn(4)},reward_mean_std={'x':torch.randn(1)},epoch=6000,frame=786432000)


def test_preserve_every_learned_state_and_original():
    w=checkpoint();before=deepcopy(w['relation_metadata']);new=migrate_push_only_checkpoint(w)
    changed={'relation_experiment_config','open_push_contract','relation_metadata'}
    for key in w:
        if key not in changed:assert new[key] is w[key]
    assert new['open_push_contract']['reward']['contact_at']['max_tilt']==.35
    assert w['relation_metadata']==before
    assert 'max_tilt' not in w['open_push_contract']['reward']['contact_at']


@pytest.mark.parametrize('kind',['input','manifest','reward','network','amp','bank'])
def test_reject_other_changes(kind):
    w=checkpoint()
    if kind=='input':w['relation_experiment_config']['observation_size']=684
    if kind=='manifest':w['relation_experiment_config']['env']['motion_file']='tokenhsi/data/dataset_open_push_push180.yaml'
    if kind=='reward':w['relation_experiment_config']['env']['relationReward']['contact_at']['progress_delta']=.1
    if kind=='network':w['relation_experiment_config']['train']['params']['network']['transformer']['num_layers']=1
    if kind=='amp':w['amp_enabled']=False
    if kind=='bank':w['open_push_contract']['rsi_bank_sha256']='bad'
    with pytest.raises(ValueError):migrate_push_only_checkpoint(w)


def test_physical_push_runtime_gate_and_dense_reward():
    from env.tasks.multi_agent.open_push_reward import PushOnlyRuntime
    a,b=configs();roots,hands,boxes,goals,start=geometry()
    rt=PushOnlyRuntime(1,b['relationGraph'],b['relationReward'],1/30,'cpu',b['pushBox']['size'],True)
    ids=torch.arange(1);rt.reset(ids,torch.ones(1,2,dtype=torch.long),boxes)
    boxes[...,:3]=goals
    def step():return rt.step(roots,hands,None,None,boxes,goals)
    assert step()['success'].all()
    boxes[...,3]=torch.sin(torch.tensor(.4/2));boxes[...,6]=torch.cos(torch.tensor(.4/2))
    assert not step()['success'].any() and not rt.done.any()
    boxes[...,:3]=start
    old=contact_at_reward(roots,hands,boxes,goals,a['relationReward'],a['pushBox']['size'],None,True)
    new=contact_at_reward(roots,hands,boxes,goals,b['relationReward'],b['pushBox']['size'],None,True)
    for key in old:torch.testing.assert_close(old[key],new[key])
