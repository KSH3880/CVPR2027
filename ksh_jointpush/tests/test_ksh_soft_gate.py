import torch
import yaml
from pathlib import Path
from ksh_jointpush.ksh_soft_gate import soft_group_gate,SoftStage2JointPushRuntime
from utils.stage2_jointpush import compose_graph,group_contact_gate


def test_one_ready_member_earns_half_credit_without_other_groups_affecting_it():
    boxes=torch.zeros(1,4,13);boxes[...,6]=1
    sizes=torch.tensor([4.,2.,1.5]).expand(1,4,3)
    goals=torch.tensor([0.,5.,0.]).expand(1,4,3)
    groups=torch.tensor([[0,0,2,3]])
    hands=torch.zeros(1,4,2,3);hands[...,1]=-1
    hands[0,1,:,1]=-4
    score=soft_group_gate(hands,boxes,sizes,goals,groups)
    torch.testing.assert_close(score,torch.tensor([[.5,.5,1.,1.]]))
    assert not group_contact_gate(hands,boxes,sizes,goals,groups)[0,:2].any()
    hands[0,1,:,1]=-1.18
    moved=soft_group_gate(hands,boxes,sizes,goals,groups)
    assert (moved[0,:2]>.5).all() and (moved[0,:2]<1).all()
    hands[0,1,:,1]=-1
    torch.testing.assert_close(soft_group_gate(hands,boxes,sizes,goals,groups),torch.ones(1,4))


def make_runtime(environments=1):
    root=Path(__file__).resolve().parents[2]
    config=yaml.safe_load((root/'ksh_jointpush/tokenhsi/data/cfg/multi_agent/stage2_jointpush.yaml').read_text())['env']
    groups=torch.tensor([[0,0,2,3]]).expand(environments,-1)
    graph=compose_graph(torch.full((environments,4),5),groups,False)
    return SoftStage2JointPushRuntime(environments,graph,config['relationReward'],'cpu',contact_weights=[.2]*3),graph


def test_partial_contact_still_shapes_before_goal_without_false_success():
    runtime,graph=make_runtime()
    runtime.gate[0,:2]=False;runtime.soft_gate[0,:2]=.5;runtime.contact[:]=True
    phi=torch.full((1,8),.5);progress=torch.ones_like(phi);zero=torch.zeros_like(phi)
    result=runtime.step(phi,progress,zero,zero,zero)
    at=(graph.edge_relation==7)&(graph.edge_owner<2)
    push=graph.edge_relation!=7
    torch.testing.assert_close(result['state_component'][at],torch.full((2,),.05))
    torch.testing.assert_close(result['progress_component'][at],torch.full((2,),.1))
    assert not result['own_success'][at].any() and (result['success_component'][at]==0).all()
    torch.testing.assert_close(result['total'][push],torch.full((4,),.6))
    assert not runtime.done.any()


def test_box_goal_success_saturates_both_edges_even_with_no_hand_contact():
    runtime,graph=make_runtime()
    runtime.gate[:]=False;runtime.soft_gate[:]=0;runtime.contact[:]=False
    at=graph.edge_relation==7
    phi=at.float();zero=torch.zeros_like(phi)
    result=runtime.step(phi,zero,zero,zero,zero)
    assert runtime.done.all() and result['own_success'][at].all()
    assert not result['own_success'][~at].any()
    for component in ('state_component','progress_component','success_component'):
        torch.testing.assert_close(result[component],torch.full((1,8),.2))
    torch.testing.assert_close(result['agent_task_reward'],torch.full((1,4),1.2))
    torch.testing.assert_close(result['group_gate'],torch.ones(1,4))
    assert (result['soft_group_gate_raw']==0).all()
    # Leaving the goal cancels saturation; this is current-state success, not a latch.
    result=runtime.step(torch.zeros_like(phi),zero,zero,zero,zero)
    assert not runtime.done.any()
    assert (result['agent_task_reward']==0).all()


def test_goal_success_keeps_height_tolerance_and_partial_reset_isolated():
    runtime,graph=make_runtime(2)
    runtime.gate[:]=False;runtime.soft_gate[:]=0
    at=graph.edge_relation==7
    phi=at.float();zero=torch.zeros_like(phi)
    z=zero.clone();z[1,at[1]]=.01
    runtime.step(phi,zero,z,zero,zero)
    assert runtime.done[0].all() and not runtime.done[1].any()
    runtime.reset(torch.tensor([0]),phi[:1],zero[:1],zero[:1],zero[:1])
    assert runtime.done[0].all() and not runtime.done[1].any()
