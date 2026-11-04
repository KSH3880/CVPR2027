from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_continuous_progress import ContinuousProgress
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from test_before_asymmetric_progress import geometry, ENV

NAME='stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_radius3m_continuous'

def run(c,d,gate,wait=None):
    d=torch.tensor(d,dtype=torch.float32).reshape(c.seen.shape)
    q=1/(1+d)
    w=1/(1+(d-3).clamp_min(0)+10*(3-d).clamp_min(0)) if wait is None else wait
    return c.step(q,w,torch.full_like(c.seen,gate),torch.ones_like(c.seen))

def test_stationary_open_approach_retreat_and_reopen():
    c=ContinuousProgress((1,1),'cpu')
    before=run(c,[3],False);opened=run(c,[3],True)
    assert torch.equal(before,opened) and opened.item()==.5
    assert torch.equal(run(c,[3],True),opened)
    near=run(c,[1],True);assert .5<near.item()<1
    assert run(c,[2],True).item()<near.item()
    assert torch.equal(run(c,[1],False),near)
    assert torch.equal(run(c,[1],True),near)
    assert run(c,[0],True).item()==1

@pytest.mark.parametrize('d',[.1,1.,2.9,3.,4.,8.])
def test_opening_continuous_inside_and_outside(d):
    c=ContinuousProgress((1,1),'cpu');closed=run(c,[d],False)
    assert torch.equal(run(c,[d],True),closed)
    for _ in range(10):assert torch.equal(run(c,[d],True),closed)
    assert run(c,[0],True).item()==1

def test_partial_reset_and_exact_goal():
    c=ContinuousProgress((2,1),'cpu');run(c,[3,4],True)
    q0=c.q0.clone();c.reset(torch.tensor([0]))
    assert not c.seen[0].any() and c.seen[1].all() and torch.equal(c.q0[1],q0[1])
    c=ContinuousProgress((1,1),'cpu');r=run(c,[0],False)
    assert torch.equal(run(c,[0],True),r)
    assert torch.isfinite(run(c,[1],True)).all()

@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_runtime_preserves_state_success_and_no_saturation_jump(family):
    args,rows,edges,_=geometry(family,[0.,0.,0.,0.]);graph=args[-2];cfg=args[-1]
    common=dict(contact_weights=ENV['jointPush']['contact_weights'])
    rt=BeforePushRuntime(4,graph,cfg,'cpu',continuous_progress=True,**common)
    orig=BeforePushRuntime(4,graph,cfg,'cpu',**common)
    phi=torch.ones_like(rt.phi);zero=torch.zeros_like(phi);progress=torch.full_like(phi,.25)
    baseline=orig.step(phi,progress,zero,zero,zero)
    rt.progress_inputs=(progress,torch.ones_like(phi))
    # Diagnostics must not capture the opening.
    rt.components(phi,progress,zero,zero,zero,graph);assert not rt.continuous_progress.seen.any()
    result=rt.step(phi,progress,zero,zero,zero)
    for key in ['state_component','success_component','own_success','prerequisite_gate']:
        assert torch.equal(result[key],baseline[key])
    dep=graph.prereq_mask.any(-1)
    assert (result['progress_component'][dep]==.1).all()
    assert torch.equal(result['progress_component'][~dep],baseline['progress_component'][~dep])
    rt.progress_inputs=(torch.ones_like(phi),torch.ones_like(phi))
    result=rt.step(phi,progress,zero,zero,zero)
    assert (result['progress_component'][dep]==.2).all()
    rt.reset(torch.tensor([0]),phi[:1],zero[:1],zero[:1],zero[:1])
    assert not rt.continuous_progress.seen[0].any() and rt.continuous_progress.seen[1:].any()

def test_config_contract():
    cfg=native_config(NAME);env=cfg['env'];validate_unified_env(env)
    validate_experiment(NAME,env,native_config(NAME,True))
    assert 'approach_timeout' not in env['before'] and 'state_rescale' not in env['before']
    base=native_config(NAME[:-len('_continuous')])['env']
    copy=deepcopy(env);copy['before'].pop('continuous_progress');assert copy==base
