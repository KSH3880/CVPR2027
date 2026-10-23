"""PUSH-only configuration has no inactive OPEN objective or reference data."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
import torch
import yaml
from env.tasks.multi_agent.open_push_reward import OpenPushRuntime,PushOnlyRuntime
from utils.open_push_spec import validate_env
from utils.push_only import project_bank

CONFIG=Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml')


def test_no_open_settings_or_motion_assets_and_identical_push_candidates():
    env=yaml.safe_load(CONFIG.read_text())['env'];validate_env(env)
    def check(value):
        if isinstance(value,dict):
            for key,item in value.items():
                assert 'door' not in key.lower() and key!='OPEN'
                check(item)
        elif isinstance(value,list):
            for item in value:check(item)
        elif isinstance(value,str):assert 'door' not in value.lower()
    check(env)
    motions=yaml.safe_load(Path('tokenhsi/data/dataset_push_only_push180.yaml').read_text())
    check(motions);assert set(motions['motions'])=={'loco','push'}
    bank=json.loads(Path(env['openPushRSI']['bank_file']).read_text());check(bank)
    source=json.loads(Path('tokenhsi/data/open_push_contact_at_friction06_random_size_goal1to25_bank.json').read_text())
    assert bank==project_bank(source)
    assert len(bank['candidates']['push'])==len(source['candidates']['push'])
    bad=deepcopy(env);bad['door']={}
    with pytest.raises(ValueError,match='door settings'):validate_env(bad)


def test_push_only_runtime_exactly_preserves_push_rewards_and_success():
    env=yaml.safe_load(CONFIG.read_text())['env']
    legacy=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01.yaml').read_text())['env']
    n=32
    old=OpenPushRuntime(n,legacy['relationGraph'],legacy['relationReward'],legacy['doorHold'],1/30,'cpu',[1.3]*3,True)
    new=PushOnlyRuntime(n,env['relationGraph'],env['relationReward'],1/30,'cpu',[1.3]*3,True)
    ids=torch.arange(n);tasks=torch.ones(n,2,dtype=torch.long)
    boxes=torch.zeros(n,2,13);boxes[...,2]=.65;boxes[...,6]=1
    old.reset(ids,tasks,boxes);new.reset(ids,tasks,boxes)
    torch.manual_seed(7)
    roots=torch.randn(n,2,3);hands=torch.randn(n,2,2,3)
    goals=boxes[...,:3].clone();goals[...,:2]+=torch.randn(n,2,2)*.5
    for contact in (False,True):
        old.push_contact[:]=contact;new.push_contact[:]=contact
        a=old.step(roots,hands,torch.zeros_like(roots),torch.zeros(n,2),boxes,goals)
        b=new.step(roots,hands,None,None,boxes,goals)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(old.phi,new.phi,rtol=0,atol=0)
        assert torch.equal(old.own_success,new.own_success)
    assert not hasattr(new,'hold') and not hasattr(new,'open_credit')
