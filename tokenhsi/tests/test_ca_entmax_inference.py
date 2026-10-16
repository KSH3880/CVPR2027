import pytest
import torch

from learning.multi_agent.ca_entmax_inference import entmax15, enable_ca_entmax
from learning.multi_agent.task_coordination import TaskCoordination


def test_entmax_matches_independent_bisection_with_masked_tasks():
    torch.manual_seed(17)
    scores = torch.randn(64, 6, dtype=torch.float64) * 3
    scores[:32, 3:] = -torch.inf
    scores[0, 1:] = -torch.inf
    values = scores / 2
    high = values.amax(-1, keepdim=True)
    low = high - 1
    for _ in range(80):
        middle = (low + high) / 2
        mass = (values - middle).clamp_min(0).square().sum(-1, keepdim=True)
        low = torch.where(mass > 1, middle, low)
        high = torch.where(mass > 1, high, middle)
    reference = (values - (low + high) / 2).clamp_min(0).square()
    actual = entmax15(scores)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual.sum(-1), torch.ones(64, dtype=torch.float64))
    assert (actual[:32, 3:] == 0).all()


@pytest.mark.parametrize('people', [2, 6])
def test_uniform_attention_preserves_original_block_output(people):
    torch.manual_seed(17)
    coordination = TaskCoordination(num_layers=4, feedforward_dim=128)
    block = coordination.blocks[0]
    with torch.no_grad():
        block.attention.in_proj_weight[:128].zero_()
        block.attention.in_proj_bias[:128].zero_()
    humans, tokens = torch.randn(3, people, 64), torch.randn(3, 6, 64)
    relations = torch.zeros(3, people, 6, dtype=torch.long)
    padding = torch.zeros(3, 1, 1, 6, dtype=torch.bool)
    padding[0, ..., -1] = True
    original = block(humans, tokens, relations, padding)
    weights = {key: value.clone() for key, value in coordination.state_dict().items()}
    enable_ca_entmax(coordination)
    torch.testing.assert_close(block(humans, tokens, relations, padding), original, atol=1e-6, rtol=1e-5)
    for key, value in coordination.state_dict().items():
        assert torch.equal(weights[key], value)


def test_ca_task_permutation_and_masked_token_invariance():
    torch.manual_seed(17)
    coordination = TaskCoordination(num_layers=4, feedforward_dim=128)
    enable_ca_entmax(coordination)
    block = coordination.blocks[0]
    humans, tokens = torch.randn(2, 6, 64), torch.randn(2, 6, 64)
    relations = torch.randint(3, (2, 6, 6))
    padding = torch.zeros(2, 1, 1, 6, dtype=torch.bool)
    padding[..., -1] = True
    expected = block(humans, tokens, relations, padding)
    order = torch.tensor([3, 0, 5, 2, 4, 1])
    actual = block(humans, tokens[:, order], relations[:, :, order], padding[..., order])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    tokens[:, -1] += 1000
    torch.testing.assert_close(block(humans, tokens, relations, padding), expected, atol=2e-6, rtol=1e-5)
