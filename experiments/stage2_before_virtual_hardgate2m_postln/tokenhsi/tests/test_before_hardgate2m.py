from copy import deepcopy
import pytest
import torch
from env.tasks.multi_agent.before_reward import predecessor_at_progress
from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.edge_ontop_spec import permute_graph, select_graph
from test_before_asymmetric_progress import geometry

NAME='stage2_before_virtual_hardgate2m_postln'
BASE='stage2_before_virtual_asym_postln'
ENV=native_config(NAME)['env']


def test_config_and_checkpoint_contract():
    cfg=native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME,cfg['env'],native_config(NAME,True))
    before=deepcopy(cfg['env']); old=deepcopy(native_config(BASE)['env'])
    assert before['before'].pop('hard_gate_progress')=={'distance':2.}
    assert old['before'].pop('asymmetric_progress')=={'lambda_inner':10.}
    old['numEnvs']=before['numEnvs']
    assert before==old
    assert cfg['sim']==native_config(BASE)['sim']
    for source,target in ((ENV,native_config(BASE)['env']),(native_config(BASE)['env'],ENV)):
        with pytest.raises(ValueError,match='checkpoint contract'):
            check_checkpoint({'before_contract':checkpoint_contract(source)},target)


@pytest.mark.parametrize('value',[-1.,0.,float('inf'),float('nan'),True,'2',None])
def test_invalid_threshold(value):
    env=deepcopy(ENV);env['before']['hard_gate_progress']['distance']=value
    with pytest.raises(ValueError,match='Hard gate progress'):
        validate_unified_env(env)


@pytest.mark.parametrize('family',['sit','climb','stack'])
def test_threshold_release_runtime_and_batch(family):
    args,rows,edges,radius=geometry(family,[0.]*4)
    roots,boxes,sizes,goals,g,config=args
    distance=torch.tensor([1.999,2.,2.001,0.])
    source=g.edge_src[rows,edges]
    if family=='stack': boxes[rows,source-4,0]=distance
    else: roots[rows,source,0]=distance
    original=torch.rand(4,len(g.ids)); saved=original.clone()
    raw=torch.zeros_like(original,dtype=torch.bool)
    old=predecessor_at_progress(original,*args)
    actual=predecessor_at_progress(original,*args,raw_success=raw,hard_gate_distance=2.)
    expected=old.clone();expected[rows[[0,3]],edges[[0,3]]]=0
    torch.testing.assert_close(actual,expected)
    assert torch.equal(original,saved)
    # XY distance; goal height is irrelevant. ON_TOP uses the carried object.
    high=goals.clone();high[...,2]+=100
    torch.testing.assert_close(predecessor_at_progress(original,roots,boxes,sizes,high,g,config,
        raw_success=raw,hard_gate_distance=2.),actual)
    leader=g.prereq_mask[rows,edges].long().argmax(-1)
    raw[rows,leader]=True
    torch.testing.assert_close(predecessor_at_progress(original,*args,raw_success=raw,hard_gate_distance=2.),old)
    runtime=BeforePushRuntime(4,g,config,'cpu',contact_weights=ENV['jointPush']['contact_weights'])
    for released in (False,True,False):
        phi=torch.full_like(original,.1);z=torch.ones_like(original);errors=torch.ones_like(original)
        phi[rows,leader]=1. if released else .1;z[rows,leader]=0.
        raw=interaction_own_success(phi,z,errors,g,config,errors)
        value=predecessor_at_progress(original,*args,raw_success=raw,hard_gate_distance=2.)
        result=runtime.components(phi,value,z,errors,errors,g)
        previous=runtime.components(phi,old,z,errors,errors,g)
        assert (result['prerequisite_gate'][rows,edges]==released).all()
        torch.testing.assert_close(result['progress_component'][rows,edges],.2*value[rows,edges])
        for key in ('state_component','success_component','own_success','reward_saturated'):
            assert torch.equal(result[key],previous[key])
        order=torch.rand_like(original).argsort(-1)
        shuffled=predecessor_at_progress(original.gather(1,order),roots,boxes,sizes,goals,
            permute_graph(g,order),config,raw_success=raw.gather(1,order),hard_gate_distance=2.)
        torch.testing.assert_close(shuffled,value.gather(1,order))
        ids=torch.tensor([3,1])
        partial=predecessor_at_progress(original[ids],roots[ids],boxes[ids],sizes[ids],goals[ids],
            select_graph(g,ids),config,raw_success=raw[ids],hard_gate_distance=2.)
        torch.testing.assert_close(partial,value[ids])
