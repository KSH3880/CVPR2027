import copy

import torch

from learning.multi_agent.ca_solo_task_mask_inference import enable_solo_task_mask
from learning.multi_agent.task_coordination import TaskCoordination, task_relations


def test_matches_toy_and_preserves_solo_queries_and_permutations():
    torch.manual_seed(42)
    model = TaskCoordination(num_layers=4, feedforward_dim=128)
    original = copy.deepcopy(model)
    enable_solo_task_mask(model)
    packet = torch.tensor([[1, 2, 0, 6, 14], [1, 2, 1, 6, 14],
                           [1, 2, 2, 7, 16], [1, 2, 3, 7, 16],
                           [1, 2, 4, 8, 18], [1, 3, 5, 9, 10]])[None]
    relations = task_relations(packet, 6)
    humans, tokens = torch.randn(1, 6, 64), torch.randn(1, 6, 64)
    padding = torch.zeros(1, 1, 1, 6, dtype=torch.bool)
    oracle_mask = padding.expand(1, 1, 6, 6).clone()
    oracle_mask[:, :, :4, 4:] = True
    order = torch.tensor([4, 1, 3, 0, 5, 2])
    for block, reference in zip(model.blocks, original.blocks):
        baseline = reference(humans, tokens, relations, padding)
        expected = reference(humans, tokens, relations, oracle_mask)
        actual = block(humans, tokens, relations, padding)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 4:], baseline[:, 4:], atol=0, rtol=0)
        permuted = block(humans[:, order], tokens[:, order],
                         relations[:, order][:, :, order], padding[..., order])
        torch.testing.assert_close(permuted, actual[:, order], atol=1e-6, rtol=1e-5)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original.state_dict()[key])


def test_no_solo_tasks_or_no_joint_queries_leaves_output_unchanged():
    model = TaskCoordination(num_layers=4, feedforward_dim=128)
    original = copy.deepcopy(model)
    enable_solo_task_mask(model)
    humans, tokens = torch.randn(1, 2, 64), torch.randn(1, 2, 64)
    padding = torch.zeros(1, 1, 1, 2, dtype=torch.bool)
    for relations in [torch.tensor([[[1, 2], [2, 1]]]), torch.tensor([[[1, 0], [0, 1]]])]:
        for block, reference in zip(model.blocks, original.blocks):
            torch.testing.assert_close(block(humans, tokens, relations, padding),
                                       reference(humans, tokens, relations, padding), atol=0, rtol=0)
