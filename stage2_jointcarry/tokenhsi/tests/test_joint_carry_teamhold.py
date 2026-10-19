from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent.joint_carry_reward import (
    JointCarryRuntime, apply_joint_geometry, joint_holding_score, opposite_gate)
from utils.edge_context_spec import HOLDING
from utils.edge_ontop_spec import permute_graph
from utils.joint_carry_spec import validate_env
from utils.joint_carry_amp import validate_ablation
from utils.mixed_carry_spec import sample_graph, joint_scenes
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from test_joint_carry_amp_ablation import configs


def test_config_changes_only_joint_holding_state():
    base, old_train = configs('locoamp_ca4_goal360_task_embedding')
    env, train = configs('locoamp_ca4_goal360_teamhold_task_embedding')
    validate_env(env);validate_ablation(env,train);validate_relation_config(env['relationReward'])
    restored=deepcopy(env)
    assert restored['relationReward']['joint_carry'].pop('holding_state')=='opposite_anchor_min'
    assert restored==base
    restored=deepcopy(train)
    restored['params']['config']['name']=old_train['params']['config']['name']
    assert restored==old_train
    for source,target in [(base,env),(env,base)]:
        with pytest.raises(ValueError):
            check_checkpoint_metadata({'relation_metadata':checkpoint_metadata(source['relationReward'])},
                                      checkpoint_metadata(target['relationReward']))
    bad=deepcopy(env['relationReward']);bad['joint_carry']['holding_state']='min_individual'
    with pytest.raises(ValueError,match='holding state'):
        validate_relation_config(bad)


def test_pair_assignment_bottleneck_and_gate():
    scores=torch.tensor([[[.9,.1],[.1,.3]],[[1.,.1],[.1,.3]],
                         [[.9,.1],[.1,.7]],[[.9,.1],[.1,.9]],
                         [[1.,.1],[1.,.1]],[[.1,.9],[.8,.1]]])
    c=joint_holding_score(scores)
    torch.testing.assert_close(c,torch.tensor([.3,.3,.7,.9,.1,.8]))
    assert torch.equal(c>=.9,opposite_gate(scores,.9))
    torch.testing.assert_close(c,joint_holding_score(scores.flip(1)))
    torch.testing.assert_close(c,joint_holding_score(scores.flip(2)))


def geometry(env,graph,hands):
    n=len(hands)
    objects=torch.zeros(n,4,13);objects[...,6]=1.
    sizes=torch.tensor([.52,.8,.4]).expand(n,4,3)
    phi=torch.full((n,4),.95)
    diag={'target':torch.zeros(n,4,3),'distance':torch.zeros(n,4)}
    phi,gate=apply_joint_geometry(phi,diag,hands,objects,sizes,graph,env['relationReward'])
    return phi,gate,diag


@pytest.mark.parametrize('preset',['joint_carry_at','joint_carry_ontop'])
def test_only_shared_state_changes_progress_success_gate_and_independent_preserved(preset):
    base,_=configs('locoamp_ca4_goal360_task_embedding')
    env,_=configs('locoamp_ca4_goal360_teamhold_task_embedding')
    graph=sample_graph(4,env['relationGraph'],preset=preset,joint_mask=torch.tensor([True,True,True,False]))
    hands=torch.zeros(4,2,2,3)
    hands[:,0,:,1]=-.25;hands[:,1,:,1]=.25
    hands[1,1,:,1]=.55
    hands[2,1,:,1]=-.25
    phi,gate,diag=geometry(env,graph,hands)
    old_phi,old_gate,_=geometry(base,graph,hands)
    torch.testing.assert_close(phi,old_phi,rtol=0,atol=0)
    assert torch.equal(gate,old_gate) and gate.tolist()==[True,False,False,False]
    old=JointCarryRuntime(4,graph,base['relationReward'],'cpu')
    new=JointCarryRuntime(4,graph,env['relationReward'],'cpu')
    old.coupled_holding.copy_(gate);new.coupled_holding.copy_(gate)
    new.joint_holding_state.copy_(diag['joint_holding_state'])
    progress=torch.rand(4,4);zero=torch.zeros(4,4)
    before=old.step(phi,progress,zero,zero,zero)
    after=new.step(phi,progress,zero,zero,zero)
    shared=(graph.edge_relation==HOLDING)&graph.edge_valid&joint_scenes(graph)[:,None]
    expected=torch.where(shared,diag['joint_holding_state'][:,None]*.2,before['state_component'])
    torch.testing.assert_close(after['state_component'],expected)
    for key in ['phi_raw','progress_raw','own_success','success_component','progress_component']:
        torch.testing.assert_close(after[key],before[key],rtol=0,atol=0)
    torch.testing.assert_close(after['total'],before['total']+expected-before['state_component'])
    torch.testing.assert_close(after['agent_task_reward'][3],before['agent_task_reward'][3],rtol=0,atol=0)
    holding_success=after['own_success'][1][graph.edge_relation[1]==HOLDING]
    assert holding_success.sum()==1
    assert after['state_component'][1,shared[1]].min()>0  # Dense even while gate is closed.
    assert after['total'][1:3][graph.edge_relation[1:3]!=HOLDING].count_nonzero()==0
    assert torch.equal(new.done,old.done)
    order=torch.rand(4,4).argsort(-1)
    permuted=JointCarryRuntime(4,permute_graph(graph,order),env['relationReward'],'cpu')
    permuted.coupled_holding.copy_(gate);permuted.joint_holding_state.copy_(new.joint_holding_state)
    other=permuted.step(phi.gather(1,order),progress.gather(1,order),zero,zero,zero)
    torch.testing.assert_close(other['agent_task_reward'],after['agent_task_reward'])
    permuted_graph=deepcopy(graph)
    permuted_graph.edge_owner.copy_(1-graph.edge_owner)
    phi_swap,gate_swap,diag_swap=geometry(env,permuted_graph,hands.flip(1))
    torch.testing.assert_close(diag_swap['joint_holding_state'],diag['joint_holding_state'])
    assert torch.equal(gate_swap,gate)
