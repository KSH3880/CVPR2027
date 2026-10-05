"""Exercise the real Carry PPO update without loading IsaacGym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


class Observation:
    state = None
    path_progress = None

    def index(self, index):
        return self


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(0.0))

    def evaluate(self, observation, actions):
        distribution = torch.distributions.Normal(self.mean, 1.0)
        return distribution.log_prob(actions), distribution.entropy(), self.mean.expand_as(actions), None

    def all_mean_outputs(self, observation):
        return self.mean

    def diversity(self, observation, output):
        invalid = output * float('nan')
        return dict(smoothness_loss=invalid, speed_smoothness_loss=invalid)


def analytic(output, *args, **kwargs):
    invalid = output * float('nan')
    return dict(loss=invalid, curvature_loss=invalid, per_sample_loss=invalid,
                active_fraction=torch.tensor(1.0), min_hh=0.0,
                min_bb_margin=0.0, min_hb_margin=0.0)


def path_regularization(output, *args, **kwargs):
    fields = dict(consistency_loss=output * float('nan'),
                  excess_length_loss=output * float('nan'), direction_loss=output * float('nan'))
    for name in ('safe_weight', 'mean_replan_displacement', 'mean_future_length_ratio',
                 'mean_future_excess_m', 'mean_direction_error_deg', 'direction_active_fraction',
                 'direction_fallback_fraction', 'max_future_excess_m'):
        fields[name] = 0.0
    return fields


class RewardOnlyTest(unittest.TestCase):
    def test_disabled_auxiliaries_never_affect_backward_and_kl_is_post_step(self):
        source = ROOT / 'TokenHSI-coord/carry_planner/train_closed_loop.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == '_ppo_update')
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = dict(torch=torch, F=F, carry_analytic_collision_loss=analytic,
                         carry_path_regularization=path_regularization)
        exec(compile(module, str(source), 'exec'), namespace)
        policy = Policy()
        actions = torch.ones(4)
        old = torch.distributions.Normal(0.0, 1.0).log_prob(actions)
        kwargs = {arg.arg: 0.0 for arg in function.args.args}
        kwargs.update(policy=policy, optimizer=torch.optim.SGD(policy.parameters(), lr=0.05),
                      observations=Observation(), actions=actions, old_log_prob=old,
                      returns=torch.zeros(4), advantages=torch.ones(4), epochs=1,
                      minibatch=4, clip_ratio=0.15, value_coef=0.5, entropy_coef=0.0001,
                      regularization_scale=1.0)
        result = namespace['_ppo_update'](**kwargs)
        self.assertTrue(torch.isfinite(policy.mean))
        self.assertAlmostEqual(policy.mean.item(), 0.05, places=6)
        self.assertEqual(result['approx_kl_before_mean'], 0.0)
        logratio = torch.distributions.Normal(policy.mean.detach(), 1.0).log_prob(actions) - old
        expected = (logratio.exp() - 1 - logratio).mean().item()
        self.assertAlmostEqual(result['approx_kl_after_mean'], expected, places=6)
        self.assertEqual(result['approx_kl_after_mean'], result['approx_kl_after_max'])
        self.assertEqual(result['ppo_updates'], 1)


if __name__ == '__main__':
    unittest.main()
