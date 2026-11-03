import pytest
import torch
import torch.nn.functional as F

from learning.multi_agent.human_coordination import CoordinationBlock


@pytest.mark.parametrize('batch', [1, 1024])
@pytest.mark.parametrize('mode,relations_count', [('all', 4), ('before_only_mask_none', 3)])
def test_eval_float_relation_mask_matches_functional_attention(batch, mode, relations_count):
    torch.manual_seed(19)
    block = CoordinationBlock(16, 2, 32, relations_count, bias_mode=mode)
    with torch.no_grad():
        block.relation_bias.normal_()
    humans = torch.randn(batch, 4, 16)
    relations = torch.randint(relations_count, (batch, 4, 4))
    relations[:, torch.arange(4), torch.arange(4)] = 1
    padding = torch.zeros((), dtype=torch.bool)
    bias = block.effective_relation_bias()[:, relations].permute(1, 0, 2, 3)
    if mode != 'all':
        padding = padding | (relations == 0)[:, None]
    bias = bias.masked_fill(padding, float('-inf')).flatten(0, 1)
    query = block.query_norm(humans).transpose(0, 1)
    attn = block.attention
    context, _ = F.multi_head_attention_forward(query, query, query, attn.embed_dim, attn.num_heads,
        attn.in_proj_weight, attn.in_proj_bias, attn.bias_k, attn.bias_v, attn.add_zero_attn,
        attn.dropout, attn.out_proj.weight, attn.out_proj.bias, training=False,
        attn_mask=bias, need_weights=False)
    residual = humans + context.transpose(0, 1)
    expected = residual + block.ffn(block.ffn_norm(residual))
    block.eval()
    with torch.no_grad():
        actual = block(humans, None, relations, torch.zeros((), dtype=torch.bool))
    torch.testing.assert_close(actual, expected)
    block.train()
    trained = block(humans, None, relations, torch.zeros((), dtype=torch.bool))
    torch.testing.assert_close(trained, expected)
    trained.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert block.relation_bias.grad.abs().sum() > 0
