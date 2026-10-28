from types import SimpleNamespace
import torch
from ksh_jointpush.ksh_agent_rsi import sample_agent_initialization, mixed_body_overlap, install
from utils.stage2_jointpush_rsi import sample_push_initialization


def test_agent_sampling_is_independent_and_masked():
    torch.manual_seed(42)
    groups=torch.zeros(20000,4,dtype=torch.long)
    templates=torch.full_like(groups,5)
    selected=sample_agent_initialization(groups,templates)
    assert .48 < selected.float().mean() < .52
    mixed=selected.any(-1)&~selected.all(-1)
    assert .86 < mixed.float().mean() < .89
    templates[:,1]=1
    assert not sample_agent_initialization(groups,templates,1.)[:,1].any()
    assert not sample_agent_initialization(groups,templates,0.).any()


def test_legacy_group_mode_is_unchanged():
    import env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi as task
    before=(task.sample_push_initialization,task.coupled_body_overlap)
    install({'stage2JointPush':{'coupled_rsi_selection':'group'}})
    assert (task.sample_push_initialization,task.coupled_body_overlap)==before
    groups=torch.tensor([[0,0,2,2]]).expand(20000,-1)
    selected=sample_push_initialization(groups,torch.full_like(groups,5))
    assert (selected[:,0]==selected[:,1]).all()
    assert (selected[:,2]==selected[:,3]).all()
    assert .48 < selected.float().mean() < .52


def test_mixed_overlap_checks_both_agent_orders_and_partial_batch():
    body=torch.zeros(4,2,1,13);body[...,6]=1
    body[2,1,0,0]=2.
    groups=torch.zeros(4,2,dtype=torch.long)
    interaction=torch.tensor([[1,0],[0,1],[1,0],[0,0]],dtype=torch.bool)
    geometry={'ids':torch.tensor([0]),'offsets':torch.zeros(1,3),'radii':torch.tensor([.1])}
    expected=torch.tensor([True,True,False,False])
    assert torch.equal(mixed_body_overlap(body,geometry,groups,interaction),expected)
    assert torch.equal(mixed_body_overlap(body[[1,2]],geometry,groups[[1,2]],interaction[[1,2]]),expected[[1,2]])
