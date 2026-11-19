import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'FSM/dct'), str(ROOT/'FSM'), str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import isaacgym
import torch

from freeze import DelayFreeze
from state import CoordinationState
from random_delay import SETTINGS, MODELS, OUT, schedule, command


class RandomDelayTests(unittest.TestCase):
    def test_independent_release_and_fall(self):
        c = DelayFreeze.__new__(DelayFreeze)
        c.task = SimpleNamespace()
        c.delay_steps = torch.tensor([30, 61, 100, 150])[:, None]
        c.target = torch.tensor([[True, False, False], [False, True, False],
                                 [False, False, True], [True, False, False]])
        c.alive = torch.ones((4, 3), dtype=torch.bool)
        c.state = CoordinationState(torch.full((4, 3), -1), torch.zeros((4, 3)))
        c.frozen = c.target.clone()
        c.use_fsm = False
        c.trace = {k: [] for k in ('frozen', 'delayed', 'fsm_frozen', 'released')}
        c.project = lambda submit: None
        for frame in range(161):
            c.frame = frame
            if frame == 50:
                c.alive[2, 2] = False
            c.before_action()
            self.assertEqual(c.frozen.sum(1).tolist(),
                [int(frame < 30), int(frame < 61), int(frame < 50), int(frame < 150)])
        held = torch.from_numpy(np.stack(c.trace['delayed'])).sum((0, 2))
        self.assertEqual(held.tolist(), [30, 61, 50, 150])

    def test_model_pairing_and_policy_rng(self):
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        for setting in SETTINGS:
            a = schedule(setting)
            self.assertEqual(a, schedule(setting))
            self.assertEqual(len(a['steps']), 64)
            self.assertTrue(all(30 <= n <= 150 for n in a['steps']))
            paths = set()
            for model in MODELS:
                job = dict(setting=setting, model=model, seed=1, delay=0)
                cmd, env = command(job, OUT/'unused')
                paths.add(env['DCT_DELAY_SCHEDULE'])
                self.assertEqual(env['TOKENHSI_GPU'], '6')
                self.assertEqual(cmd[cmd.index('--num_envs')+1], '64')
            self.assertEqual(len(paths), 1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
