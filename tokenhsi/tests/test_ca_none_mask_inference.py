import copy

import torch

from learning.multi_agent.ca_none_mask_inference import enable_ca_none_mask
from learning.multi_agent.task_coordination import TaskCoordination, task_relations


def test_only_none_is_masked_and_weights_are_unchanged():
    torch.manual_seed(42)
    model = TaskCoordination(num_layers=4, feedforward_dim=128)
    original = copy.deepcopy(model.state_dict())
    enable_ca_none_mask(model)
    humans, tokens = torch.randn(2, 6, 64), torch.randn(2, 6, 64)
    packet = torch.tensor([[1, 2, 0, 6, 14], [1, 2, 1, 6, 14],
                           [1, 2, 2, 7, 16], [1, 2, 3, 7, 16],
                           [1, 2, 4, 8, 18], [1, 3, 5, 9, 10]])[None].repeat(2, 1, 1)
    relations = task_relations(packet, 6)
    padding = torch.zeros(2, 1, 1, 6, dtype=torch.bool)
    for block in model.blocks:
        captured = {}
        def capture(module, args, kwargs):
            captured['mask'] = kwargs['attn_mask'].reshape(2, 2, 6, 6)
        handle = block.attention.register_forward_pre_hook(capture, with_kwargs=True)
        expected = block(humans, tokens, relations, padding)
        mask = captured['mask']
        blocked = (relations == 0)[:, None].expand_as(mask)
        assert torch.isneginf(mask[blocked]).all()
        bias = block.relation_bias[:, relations].permute(1, 0, 2, 3)
        torch.testing.assert_close(mask[~blocked], bias[~blocked], atol=0, rtol=0)
        changed = tokens.clone()
        changed[:, 2:] += 100
        actual = block(humans, changed, relations, padding)
        torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=0, rtol=0)
        order = torch.tensor([4, 1, 3, 0, 5, 2])
        permuted = block(humans, tokens[:, order], relations[..., order], padding[..., order])
        torch.testing.assert_close(permuted, expected, atol=1e-6, rtol=1e-5)
        handle.remove()
    for key, value in model.state_dict().items():
        assert torch.equal(original[key], value)


def test_two_agent_joint_is_unchanged_and_empty_queries_are_finite():
    torch.manual_seed(42)
    model = TaskCoordination(num_layers=4, feedforward_dim=128)
    humans, tokens = torch.randn(3, 2, 64), torch.randn(3, 2, 64)
    relations = torch.tensor([[1, 2], [2, 1]])[None].repeat(3, 1, 1)
    padding = torch.zeros(3, 1, 1, 2, dtype=torch.bool)
    expected = humans
    for block in model.blocks:
        expected = block(expected, tokens, relations, padding)
    enable_ca_none_mask(model)
    actual = humans
    for block in model.blocks:
        actual = block(actual, tokens, relations, padding)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    relations.zero_()
    padding[0] = True
    for block in model.blocks:
        actual = block(humans, tokens, relations, padding)
        torch.testing.assert_close(actual, humans, atol=0, rtol=0)
