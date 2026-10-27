from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from utils.before_spec import validate_env, checkpoint_contract
from utils.before_full_rsi import TEMPLATE_RSI, SKILLS
from utils.size_rsi import resolve_skills
from utils.stage2_spec import check_checkpoint
from env.tasks.multi_agent.before_full_rsi_cache import BeforeFullRsiCache

ROOT=Path(__file__).resolve().parents[2]
NAME='stage2_before_ground_mix226_atprogress_fullrsi_1200'
def config(name):return yaml.safe_load((ROOT/f'tokenhsi/data/cfg/multi_agent/{name}.yaml').read_text())


def test_contract_and_stage1_probabilities():
    new=config(NAME);base=config('stage2_before_ground_mix226_atprogress_1200')
    stage1=config('stage1_v2_fixed')
    validate_env(new['env'])
    assert new['env']['templateRsi']==stage1['env']['templateRsi']==TEMPLATE_RSI
    assert new['env']['sizeAwareRsi']==stage1['env']['sizeAwareRsi']
    restored=deepcopy(new['env']);restored['before'].pop('full_rsi');restored['before']['initialization']='loco';restored['templateRsi']=base['env']['templateRsi']
    assert restored==base['env']
    assert new['sim']==base['sim']
    assert new['experiment']['transfer']['checkpoint'].endswith('00078000.pth')
    check_checkpoint({'before_contract':checkpoint_contract(new['env'])},new['env'])
    with pytest.raises(ValueError):check_checkpoint({'before_contract':checkpoint_contract(base['env'])},new['env'])
    with pytest.raises(ValueError):check_checkpoint({'before_contract':checkpoint_contract(new['env'])},base['env'])
    bad=deepcopy(new['env']);bad['sizeAwareRsi']['lateProbability']=0.
    with pytest.raises(ValueError):validate_env(bad)


def test_stage1_fallback_retains_valid_requests_and_renormalizes():
    torch.manual_seed(26);n=60000
    w=torch.tensor(TEMPLATE_RSI['HOLDING_AT']).expand(n,-1)
    available=w>0;available=available.clone();available[:,6]=False
    requested=torch.full((n,),6)
    selected,missing=resolve_skills(requested,w,available,0)
    assert missing.all() and not (selected==6).any()
    for j,p in [(0,5/7),(5,1/7),(7,1/7)]:assert abs((selected==j).float().mean().item()-p)<.01
    requested[:100]=5;selected,missing=resolve_skills(requested,w,available,0)
    assert (selected[:100]==5).all() and not missing[:100].any()


def test_late_sampling_is_seventy_percent_branch_and_shared_for_all_roles():
    # With ten valid frames, three form the late pool. Total late-frame mass is .7+.3*.3=.79.
    cache=BeforeFullRsiCache.__new__(BeforeFullRsiCache)
    n=50000;lib=SimpleNamespace(_motion_dt=torch.tensor([1.]),_motion_weights=torch.tensor([1.]),num_motions=lambda:1)
    cache.env=SimpleNamespace(device='cpu',num_envs=1,num_agents=1,num_physical_objects=1,
        _motion_lib={k:lib for k in ('sit','climb','pickUp','carryWith','putDown')},
        _before_predecessor=torch.tensor([[-1]]),_scenario_physical_object=lambda r,a:a,
        _full_rsi_frame_counts=torch.zeros(2))
    cache.spec={'lateProbability':.7};cache.sizes=[[[.5,.5,.5]]*4]
    cache.entries={(k,(10,10,10))+(((10,10,10),) if k=='putDownOnTop' else ()):
                   [(np.ones(10,dtype=bool),np.ones(10,dtype=bool))]
                   for k in ('sit','climb','pickUp','carryWith','putDown','putDownOnTop')}
    cache._prepare_sampling()
    rows=torch.zeros(n,dtype=torch.long)
    for kind,template in [('sit',1),('climb',2),('pickUp',3),('carryWith',4),('putDown',3),('putDown',4)]:
        _,times=cache.sample(kind,rows,rows,torch.full_like(rows,template))
        assert .78<(times>=7).float().mean().item()<.80
    c=cache.env._full_rsi_frame_counts
    assert .69<(c[1]/c.sum()).item()<.71


def test_predecessor_binding_selects_real_support_profile():
    cache=BeforeFullRsiCache.__new__(BeforeFullRsiCache)
    cache.env=SimpleNamespace(num_agents=4,_before_predecessor=torch.tensor([[-1,-1,0,1]]))
    cache.top_ids=torch.arange(20).reshape(1,4,5)
    rows=torch.zeros(4,dtype=torch.long);owners=torch.arange(4)
    assert cache.profile('putDownOnTop',rows,owners).tolist()==[4,9,10,16]


def test_putdown_goal_exception_preserves_physical_collision_checks():
    from env.tasks.multi_agent.shared_support_scene import scene_infeasible
    boxes=torch.zeros(1,4,13);boxes[...,6]=1
    boxes[0,:,:3]=torch.tensor([[0.,0.,.25],[0.,0.,.80],[10.,10.,.25],[12.,12.,.25]])
    body=torch.zeros(1,2,1,13);body[0,:,0,:3]=torch.tensor([[-1.,0.,1.],[1.,0.,1.]])
    t=SimpleNamespace(cfg={'env':{'before':{'full_rsi':{}}}},num_physical_objects=4,num_agents=2,
        _box_states=boxes,_box_size=torch.full((1,4,3),.5),
        _active_physical=lambda ids:torch.tensor([[True,True,False,False]]),
        _kinematic_humanoid_rigid_body_states=body,_tar_pos=torch.tensor([[[0.,0.,.25],[10.,10.,.25]]]),
        _agent_box_assignment=torch.tensor([[0,1]]),_templates=torch.tensor([[3,4]]),
        _before_predecessor=torch.tensor([[-1,0]]),_full_rsi_skill=torch.tensor([[7,7]]))
    ids=torch.tensor([0]);assert not scene_infeasible(t,ids).any()
    t._box_states[0,1,2]=.4;assert scene_infeasible(t,ids).all()
    t._box_states[0,1,2]=.8;t.cfg['env']['before']={};assert scene_infeasible(t,ids).all()


def test_airborne_support_fallback_preserves_unrelated_leader_request():
    from env.tasks.multi_agent.before_full_rsi import allocate,screen
    t=SimpleNamespace(num_envs=1,num_agents=2,device='cpu',
        cfg={'env':{'before':{'full_rsi':{'placement_retry_limit':8}}}},
        _before_predecessor=torch.tensor([[-1,0]]),_templates=torch.tensor([[3,4]]),
        _key_body_ids=torch.tensor([0,1,2,3]),
        _kinematic_humanoid_rigid_body_states=torch.ones(1,2,4,13))
    allocate(t);t._full_rsi_skill[:]=torch.tensor([[6,7]]);t._full_rsi_invalid[0,1]=True
    for _ in range(8):assert screen(t,torch.tensor([0]),lambda ids:torch.tensor([True])).all()
    assert t._full_rsi_allowed[0,0,6] # Do not turn a valid carrying leader into loco.
    assert not t._full_rsi_allowed[0,1,7]


def test_both_carry_selects_only_ontop_pairs_and_falls_back_to_loco():
    from utils.before_full_rsi import evaluation_weights
    # First row: ON_TOP pair + SIT pair; second: solo AT/TOP and a CLIMB pair.
    pre=torch.tensor([[-1,0,-1,2],[-1,-1,-1,2]])
    templates=torch.tensor([[3,4,3,1],[3,4,3,2]])
    rows=torch.arange(2).repeat_interleave(4);owners=torch.arange(4).repeat(2)
    w=torch.tensor(list(TEMPLATE_RSI.values()))[templates[rows,owners]]
    original=w.clone()
    forced=evaluation_weights(w,'both_carry',pre,templates,rows,owners)
    assert torch.equal(w,original)
    assert forced.argmax(-1).tolist()==[6,6,0,0,0,0,0,0]
    assert torch.equal(evaluation_weights(w,'mix',pre,templates,rows,owners),w)
    assert (evaluation_weights(w,'loco',pre,templates,rows,owners).argmax(-1)==0).all()
    available=torch.ones_like(forced,dtype=torch.bool);available[1,6]=False
    selected,missing=resolve_skills(forced.argmax(-1),forced,available,0)
    assert selected.tolist()==[6,0,0,0,0,0,0,0]
    assert missing.tolist()==[False,True,False,False,False,False,False,False]
