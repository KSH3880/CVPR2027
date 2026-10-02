"""Probe weights and interventions match independent PyTorch forwards."""
import torch
from torch import nn
from learning.multi_agent.attention_probe import inspect_coordination, alter_context


def test_probe_matches_real_attention_and_masked_action():
    torch.manual_seed(9)
    attention = nn.MultiheadAttention(64, 2, batch_first=True).eval()
    head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 32)).eval()
    humans, edges = torch.randn(3, 2, 64), torch.randn(3, 4, 64)
    valid = torch.tensor([[True, True, True, False], [True]*4, [True, False, False, False]])
    context, actual = attention(humans, edges, edges, key_padding_mask=~valid,
                                need_weights=True, average_attn_weights=False)
    result = inspect_coordination(attention, head, humans, edges, valid, context)
    assert result['reconstruction_error'] < 1e-6
    torch.testing.assert_close(result['attention'], actual)
    assert torch.equal(result['attention'][0, :, :, 3], torch.zeros(2, 2))
    before = head(torch.cat((humans, context), -1)).clamp(-1, 1)
    for edge in range(4):
        mask = ~valid[:2].clone()
        mask[:, edge] = True
        counter, _ = attention(humans[:2], edges[:2], edges[:2], key_padding_mask=mask)
        after = head(torch.cat((humans[:2], counter), -1)).clamp(-1, 1)
        expected = ((after-before[:2])**2).mean(-1).sqrt()
        if valid[0, edge]:
            torch.testing.assert_close(result['edge_delta'][0, :, edge], expected[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result['edge_delta'][1, :, edge], expected[1], atol=1e-6, rtol=1e-5)
    assert result['edge_delta'][2].isnan().all()


def test_edge_permutation_preserves_diagnostic_identity():
    torch.manual_seed(10)
    attention = nn.MultiheadAttention(64, 2, batch_first=True).eval()
    head = nn.Linear(128, 32).eval()
    humans, edges = torch.randn(2, 2, 64), torch.randn(2, 4, 64)
    valid = torch.ones(2, 4, dtype=torch.bool)
    order = torch.tensor([3, 0, 2, 1])
    context, _ = attention(humans, edges, edges)
    original = inspect_coordination(attention, head, humans, edges, valid, context)
    context, _ = attention(humans, edges[:, order], edges[:, order])
    changed = inspect_coordination(attention, head, humans, edges[:, order], valid[:, order], context)
    torch.testing.assert_close(changed['attention'], original['attention'][..., order])
    torch.testing.assert_close(changed['edge_delta'], original['edge_delta'][..., order], atol=1e-6, rtol=1e-5)


def test_rollout_interventions_match_reference_forwards():
    torch.manual_seed(11)
    attention = nn.MultiheadAttention(64, 2, batch_first=True).eval()
    humans, edges = torch.randn(2, 2, 64), torch.randn(2, 4, 64)
    valid = torch.tensor([[True]*4, [True, True, True, False]])
    owner = torch.tensor([[0,0,1,1],[0,0,1,0]])
    assert torch.equal(alter_context(attention,humans,edges,valid,owner,'ca_off'), torch.zeros_like(humans))
    saved = attention.in_proj_weight.detach().clone()
    with torch.no_grad():
        attention.in_proj_weight[:64].zero_()
        attention.in_proj_bias[:64].zero_()
    reference, _ = attention(humans,edges,edges,key_padding_mask=~valid)
    with torch.no_grad(): attention.in_proj_weight.copy_(saved)
    torch.testing.assert_close(alter_context(attention,humans,edges,valid,owner,'uniform'),reference)
    for human in range(2):
        mask = ~valid | (owner != human)
        reference, _ = attention(humans[:,human:human+1],edges,edges,key_padding_mask=mask)
        changed = alter_context(attention,humans,edges,valid,owner,'peer_removed')
        torch.testing.assert_close(changed[:,human:human+1],reference)
