"""AT-only fade, success boundaries and restricted RSI size-domain invariants."""
from copy import deepcopy
import json
import math
from pathlib import Path
from isaacgym import gymapi
import pytest
import torch
import yaml
from test_open_push import geometry
from env.tasks.multi_agent.open_push_reward import contact_at_reward
from utils.open_push_spec import validate_env
from utils.open_push_sizes import size_values
from utils.push_size_subset import project_size_subset

CFG=Path('tokenhsi/data/cfg/multi_agent')
BASE='open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors'
NAME='push_size120to140_at_tiltfade'

def env(name=NAME):return yaml.safe_load((CFG/(name+'.yaml')).read_text())['env']

@pytest.mark.parametrize('angle,expected',[(0,1),(5,1),(7.5,.75),(10,.5),(12.5,.25),(15,0),(30,0)])
@pytest.mark.parametrize('arrived',[False,True])
@pytest.mark.parametrize('contact',[False,True])
def test_only_at_dense_terms_fade_even_after_arrival(angle,expected,arrived,contact):
    e=env();roots,hands,boxes,goals,_=geometry()
    if arrived:boxes[...,:3]=goals
    theta=math.radians(angle);boxes[...,3]=math.sin(theta/2);boxes[...,6]=math.cos(theta/2)
    flags=torch.full((1,2),contact,dtype=torch.bool)
    old=env(BASE)['relationReward'];new=e['relationReward']
    a=contact_at_reward(roots,hands,boxes,goals,old,e['pushBox']['size'],flags,True)
    b=contact_at_reward(roots,hands,boxes,goals,new,e['pushBox']['size'],flags,True)
    at_state=torch.where(b['success'],1.,b['phi'])*expected*.2
    delta=boxes[...,:3]-goals
    distance=delta[...,:2].norm(dim=-1)
    at_motion=torch.where(b['success'],1.,1/(1+(distance-.1).clamp_min(0)))*expected*.2
    # Subtract original AT contribution: remaining PUSH must be unchanged,
    # including original arrival saturation above the new success tilt gate.
    old_at_state=torch.where(a['success'],1.,a['phi'])*.2
    old_at_motion=torch.where(a['success'],1.,1/(1+(distance-.1).clamp_min(0)))*.2
    torch.testing.assert_close(b['state'],a['state']-old_at_state+at_state,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(b['progress'],a['progress']-old_at_motion+at_motion,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(b['push_phi'],a['push_phi'])
    torch.testing.assert_close(b['push_success'],a['push_success'])
    # Success bonus has no fade multiplier. PUSH bonus retains old saturation.
    push_saturated=flags|a['success']
    torch.testing.assert_close(b['bonus'],.2*(push_saturated.float()+b['success'].float()))
    assert bool(b['success'].all())==(arrived and angle<=10)

@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('angle',[9.99,10,10.01,90,180])
def test_success_limit_and_yaw(axis,angle):
    e=env();roots,hands,boxes,goals,_=geometry();boxes[...,:3]=goals;boxes[...,3:7]=0
    boxes[...,3+axis]=math.sin(math.radians(angle)/2);boxes[...,6]=math.cos(math.radians(angle)/2)
    r=contact_at_reward(roots,hands,boxes,goals,e['relationReward'],e['pushBox']['size'],None,True)
    assert bool(r['success'].all())==(axis==2 or angle<=10)
    assert all(torch.isfinite(v).all() for v in r.values())

@pytest.mark.parametrize('changes',[{'tilt_fade_start':.4},{'tilt_fade_end':.01},{'tilt_fade_end':float('nan')},{'tilt_fade_start':True},{'max_tilt':.4}])
def test_invalid_fade(changes):
    e=env();e['relationReward']['contact_at'].update(changes)
    with pytest.raises(ValueError):validate_env(e)

def test_missing_fade_half_rejected():
    e=env();e['relationReward']['contact_at'].pop('tilt_fade_end')
    with pytest.raises(ValueError):validate_env(e)

def test_config_changes_only_at_sizes_and_bank():
    a=env(BASE);b=env();validate_env(a);validate_env(b)
    assert size_values(b)==[round(1.2+i*.01,2) for i in range(21)]
    assert len(size_values(a))==31
    for e in (a,b):
        e.pop('pushSizeRandomization');e['openPushRSI'].pop('bank_file');e['relationReward'].pop('contact_at')
    assert a==b

def test_subset_bank_contract_and_candidates():
    a=env(BASE);b=env()
    for e in (a,b):e['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
    projected=project_size_subset(a,b)
    source=json.loads(Path(a['openPushRSI']['bank_file']).read_text())
    assert projected['candidates']==source['candidates']
    assert projected==json.loads(Path(b['openPushRSI']['bank_file']).read_text())
    bad=deepcopy(b);bad['pushBox']['friction']=.7
    with pytest.raises(ValueError):project_size_subset(a,bad)


def test_old_reward_checkpoint_is_rejected():
    from utils.relation_task_spec import checkpoint_metadata,check_checkpoint_metadata
    old=checkpoint_metadata(env(BASE)['relationReward'])
    new=checkpoint_metadata(env()['relationReward'])
    with pytest.raises(ValueError,match='reward config differs'):
        check_checkpoint_metadata({'relation_metadata':old},new)
