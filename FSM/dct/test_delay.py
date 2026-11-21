import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'FSM'), str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import isaacgym
import torch
from freeze import DelayFreeze, delay_people
from state import CoordinationState


class DelayTests(unittest.TestCase):
    def test_role_selection(self):
        parent = torch.tensor([[-1, 0], [1, -1]])
        self.assertEqual(delay_people(parent).tolist(), [0, 1])
        counts = torch.bincount(delay_people(torch.full((64, 3), -1)), minlength=3)
        self.assertEqual(counts.tolist(), [22, 21, 21])

    def fixture(self, delay):
        controller = DelayFreeze.__new__(DelayFreeze)
        controller.task = SimpleNamespace()
        controller.target = torch.tensor([[True, False, False]])
        controller.alive = torch.ones((1, 3), dtype=torch.bool)
        controller.state = CoordinationState(torch.full((1, 3), -1), torch.zeros((1, 3)))
        controller.frozen = controller.target & (delay > 0)
        controller.use_fsm = False
        controller.delay_steps = delay
        controller.trace = {k: [] for k in ('frozen','delayed','fsm_frozen','released')}
        controller.project = lambda submit: None
        return controller

    def test_exact_hold_and_release_frame(self):
        c = self.fixture(120)
        for frame in range(135):
            c.frame = frame
            c.before_action()
            self.assertEqual(bool(c.frozen[0, 0]), frame < 120)
            self.assertFalse(bool(c.frozen[0, 1:].any()))
        self.assertEqual(sum(int(a.sum()) for a in c.trace['delayed']), 120)

    def test_zero_delay_and_fallen_actor(self):
        c = self.fixture(0)
        c.frame = 0
        self.assertFalse(c.before_action())
        self.assertFalse(bool(c.frozen.any()))
        c = self.fixture(120)
        c.frame = 10
        c.alive[0, 0] = False
        c.before_action()
        self.assertFalse(bool(c.frozen.any()))

    def test_fsm_wait_does_not_release_with_external_timer(self):
        c = self.fixture(0)
        c.target = torch.tensor([[True, False]])
        c.alive = torch.ones((1, 2), dtype=torch.bool)
        c.state = CoordinationState(torch.tensor([[-1, 0]]), torch.zeros((1, 2)))
        c.frozen = c.state.frozen.clone()
        c.use_fsm, c.frame = True, 0
        c.before_action()
        self.assertEqual(c.frozen.tolist(), [[False, True]])
        c.state.update(torch.tensor([[True, False]]), torch.zeros_like(c.alive), c.alive)
        self.assertFalse(bool(c.state.frozen.any()))


if __name__ == '__main__':
    unittest.main()
