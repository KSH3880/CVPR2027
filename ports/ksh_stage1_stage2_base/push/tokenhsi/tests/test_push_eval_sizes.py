import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from utils.push_eval_sizes import override_sizes, parse_range


class SizeZeroShotTests(unittest.TestCase):
    def task(self):
        task = type('HumanoidMAFiveTask', (), {})()
        task.num_agents = 2
        task._task_graph_preset = 'push_at'
        task.cfg = {'args': SimpleNamespace(eval=True), 'env': {
            'eval': {'skill': ['loco'], 'skillInitProb': [1.]},
            'pushBox': {'mass': 20., 'friction': .4}}}
        return task

    def test_centered_ranges_sampling_and_distractors_unchanged(self):
        task = self.task()
        original = torch.ones(256, 4, 3)
        env = {'PUSH_EVAL_SIZE_X': '.9,1.8', 'PUSH_EVAL_SIZE_Y': '1.7,2.3',
               'PUSH_EVAL_SIZE_Z': '1,1.6'}
        with patch.dict(os.environ, env, clear=True):
            result = override_sizes(task, original)
        for i, (lo, hi) in enumerate([(.9, 1.8), (1.7, 2.3), (1., 1.6)]):
            values = result[:, :2, i]
            self.assertGreaterEqual(float(values.min()), lo-1e-6)
            self.assertLessEqual(float(values.max()), hi+1e-6)
            self.assertGreater(len(values.unique()), 2)
        torch.testing.assert_close(result[:, 2:], original[:, 2:])
        torch.testing.assert_close(original, torch.ones_like(original))
        self.assertEqual(task.cfg['env']['pushBox'], {'mass': 20., 'friction': .4})

    def test_reject_training_rsi_and_combined_relocation(self):
        task = self.task()
        with patch.dict(os.environ, {'PUSH_EVAL_SIZE_Y': '2.3'}, clear=True):
            task.cfg['args'].eval = False
            with self.assertRaises(ValueError): override_sizes(task, torch.ones(1, 4, 3))
            task.cfg['args'].eval = True
            task.cfg['env']['eval']['skill'] = ['push']
            with self.assertRaises(ValueError): override_sizes(task, torch.ones(1, 4, 3))
            task.cfg['env']['eval']['skill'] = ['loco']
            os.environ['PUSH_BOX_SWITCH_MODES'] = 'behind_agent'
            with self.assertRaises(ValueError): override_sizes(task, torch.ones(1, 4, 3))

    def test_parse_fixed_and_reject_bad_ranges(self):
        self.assertEqual(parse_range('1.3'), [1.3, 1.3])
        for value in ('0', 'nan', '1,0.5', '1,1.03', '1,2,3'):
            with self.assertRaises(ValueError): parse_range(value)

    def test_out_of_distribution_z_keeps_explicit_push_task_identity(self):
        from utils.five_task import size_probabilities, legacy_size_probabilities
        import yaml
        with open('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04.yaml') as f:
            env = yaml.safe_load(f)['env']
        sizes = torch.tensor([[[.9, 2.3, 1.], [1.8, 1.7, 1.6]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        probs = size_probabilities(sizes, env, mask)
        torch.testing.assert_close(probs[..., -1], torch.ones(1, 2))
        self.assertEqual(float(probs[..., :-1].sum()), 0.)
        self.assertEqual(float(legacy_size_probabilities(sizes, env, mask).sum()), 0.)


if __name__ == '__main__':
    unittest.main()
