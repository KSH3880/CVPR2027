import unittest
import torch
from coordinator.task_allocation import TaskAllocationModel


class AllocationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = TaskAllocationModel(6, 5, 3, d_model=16, nhead=2).eval()
        self.agents = torch.randn(2, 2, 6)
        self.boxes = torch.randn(2, 3, 5)
        self.goals = torch.randn(2, 3, 3)

    def test_other_pairs_cannot_change_task_feature(self):
        original = self.model(self.agents, self.boxes, self.goals)
        changed = self.boxes.clone()
        changed[:, 1:] += 100
        result = self.model(self.agents, changed, self.goals + torch.tensor(
            [[[0., 0., 0.], [50., 50., 50.], [50., 50., 50.]]]))
        torch.testing.assert_close(original['task_features'][:, 0], result['task_features'][:, 0])
        own_goal = self.goals.clone()
        own_goal[:, 0] += 10
        own = self.model(self.agents, self.boxes, own_goal)
        self.assertFalse(torch.allclose(original['task_features'][:, 0], own['task_features'][:, 0]))

    def test_permutation_equivariance(self):
        original = self.model(self.agents, self.boxes, self.goals)
        p = torch.tensor([2, 0, 1])
        result = self.model(self.agents, self.boxes[:, p], self.goals[:, p])
        torch.testing.assert_close(original['logits'][:, :, p], result['logits'][:, :, :-1])
        torch.testing.assert_close(original['logits'][:, :, -1], result['logits'][:, :, -1])
        swapped = self.model(self.agents.flip(1), self.boxes, self.goals)
        torch.testing.assert_close(original['logits'].flip(1), swapped['logits'])

    def test_padding_idle_and_backward(self):
        valid = torch.tensor([[True, False, True], [False, False, False]])
        boxes = self.boxes.masked_fill(~valid[..., None], float('nan'))
        allowed = torch.ones(2, 2, 3, dtype=torch.bool)
        allowed[0, 0, 0] = False
        out = self.model(self.agents, boxes, self.goals, valid, allowed)
        probs = out['probabilities']
        self.assertTrue(torch.isfinite(probs).all())
        self.assertEqual(probs[0, 0, 0].item(), 0)
        torch.testing.assert_close(probs[1, :, -1], torch.ones(2))
        torch.testing.assert_close(probs.sum(-1), torch.ones(2, 2))
        (-probs[0, 1, 0].log() + out['task_context'].square().mean()).backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.goal_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.cross_attention.in_proj_weight.grad.abs().sum().item(), 0)
        empty = self.model(self.agents, self.boxes[:, :0], self.goals[:, :0])
        torch.testing.assert_close(empty['probabilities'], torch.ones(2, 2, 1))


if __name__ == '__main__':
    unittest.main()
