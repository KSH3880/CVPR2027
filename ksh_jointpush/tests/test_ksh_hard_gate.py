import torch
from ksh_jointpush.ksh_hard_gate import HardStage2JointPushRuntime
from ksh_jointpush.tests.test_ksh_soft_gate import make_runtime
from utils.stage2_jointpush import group_contact_gate


def hard_runtime():
    soft, graph = make_runtime()
    return HardStage2JointPushRuntime(1, graph, soft.config, 'cpu', contact_weights=[.2]*3), graph


def test_original_tolerance_requires_every_member_and_both_hands():
    boxes=torch.zeros(1,4,13);boxes[...,6]=1
    sizes=torch.tensor([4.,2.,1.5]).expand(1,4,3)
    goals=torch.tensor([0.,5.,0.]).expand(1,4,3)
    groups=torch.tensor([[0,0,2,3]])
    hands=torch.zeros(1,4,2,3);hands[...,1]=-1.17
    assert group_contact_gate(hands,boxes,sizes,goals,groups).all()
    hands[0,1,0,1]=-1.19
    gate=group_contact_gate(hands,boxes,sizes,goals,groups)
    assert not gate[0,:2].any() and gate[0,2:].all()


def test_before_goal_gate_blocks_at_but_retains_contact_reward():
    runtime,graph=hard_runtime();runtime.gate[:]=False;runtime.contact[:]=True
    phi=torch.full((1,8),.5);zero=torch.zeros_like(phi)
    result=runtime.step(phi,torch.ones_like(phi),zero,zero,zero)
    at=graph.edge_relation==7
    assert (result['total'][at]==0).all()
    torch.testing.assert_close(result['total'][~at],torch.full((4,),.6))
    runtime.gate[:]=True
    result=runtime.step(phi,torch.ones_like(phi),zero,zero,zero)
    torch.testing.assert_close(result['total'][at],torch.full((4,),.3))


def test_goal_saturates_even_with_gate_closed_and_releases_after_leaving():
    runtime,graph=hard_runtime();runtime.gate[:]=False
    phi=(graph.edge_relation==7).float();zero=torch.zeros_like(phi)
    result=runtime.step(phi,zero,zero,zero,zero)
    torch.testing.assert_close(result['agent_task_reward'],torch.full((1,4),1.2))
    assert runtime.done.all() and (result['group_gate']==1).all()
    result=runtime.step(zero,zero,zero,zero,zero)
    assert not runtime.done.any() and (result['agent_task_reward']==0).all()
