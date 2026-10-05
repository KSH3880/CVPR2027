"""CPU regression tests for PPO stopping, without importing IsaacGym."""
import ast
import copy
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
REPOS = ('TokenHSI-steer', 'TokenHSI-ma')


def load_methods(repo, filename, names):
    tree = ast.parse((ROOT / repo / 'tokenhsi/learning' / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [copy.deepcopy(n) for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = dict(torch=torch, np=np, time=time,
                     torch_ext=SimpleNamespace(
                         mean_list=lambda x: torch.stack(x).mean(),
                         policy_kl=lambda mu, sigma, old_mu, old_sigma, reduce: ((mu - old_mu) ** 2).mean() / 2))
    exec(compile(module, filename, 'exec'), namespace)
    return {name: namespace[name] for name in names}


class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(0.0))
        self.frozen = nn.Identity()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.mean.expand_as(batch['prev_actions']), torch.zeros_like(batch['prev_actions']), None, None


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a2c_network = Gaussian()

    def neglogp(self, actions, mean, sigma, logstd):
        return -torch.distributions.Normal(mean, sigma).log_prob(actions).sum(-1)

    def forward(self, batch):
        mu, logstd, _, _ = self.a2c_network(batch)
        return dict(prev_neglogp=self.neglogp(batch['prev_actions'], mu, logstd.exp(), logstd),
                    values=mu, entropy=torch.zeros_like(mu[:, 0]), mus=mu, sigmas=logstd.exp())


class TargetKLTests(unittest.TestCase):
    def agent(self, repo):
        methods = load_methods(repo, 'common_agent.py',
                               ['_post_update_approx_kl', '_approx_kl_from_neglogp',
                                '_record_ppo_minibatch_kl', '_target_kl_exceeded', '_load_config_params'])
        cls = type('Agent', (), methods)
        agent = cls()
        agent.target_kl = 0.015
        agent.mixed_precision = False
        agent.multi_gpu = False
        agent.rank = 0
        agent.epoch_num = 1
        agent._ppo_kl_step = 0
        agent.events = []
        agent.flushes = 0
        def flush():
            agent.flushes += 1
        agent.writer = SimpleNamespace(
            add_scalar=lambda name, value, step: agent.events.append((name, value, step)), flush=flush)
        agent.model = Policy()
        agent.model.train()
        agent.model.a2c_network.frozen.eval()
        return agent

    def test_post_optimizer_step_and_logprob_sign(self):
        for repo in REPOS:
            with self.subTest(repo=repo):
                agent = self.agent(repo)
                actions = torch.tensor([[-1.0], [0.0], [1.0]])
                old = -torch.distributions.Normal(0.0, 1.0).log_prob(actions).sum(-1)
                batch = {'prev_actions': actions}
                self.assertEqual(agent._post_update_approx_kl(batch, old).item(), 0.0)
                optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.1)
                optimizer.zero_grad()
                (-agent.model.a2c_network.mean).backward()
                optimizer.step()
                actual = agent._post_update_approx_kl(batch, old)
                new_logprob = torch.distributions.Normal(0.1, 1.0).log_prob(actions).sum(-1)
                logratio = new_logprob + old
                expected = ((logratio.exp() - 1) - logratio).mean()
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
                self.assertTrue(agent.model.training)
                self.assertFalse(agent.model.a2c_network.frozen.training)
                self.assertFalse(actual.requires_grad)

    def test_masks_disabled_and_invalid_kl(self):
        for repo in REPOS:
            agent = self.agent(repo)
            actions = torch.tensor([[0.0], [100.0]])
            old = -torch.distributions.Normal(0.0, 1.0).log_prob(actions).sum(-1)
            agent.model.a2c_network.mean.data.fill_(1.0)
            batch = {'prev_actions': actions}
            masked = agent._post_update_approx_kl(batch, old, torch.tensor([[1.0], [0.0]]))
            expected = torch.exp(torch.tensor(-0.5)) - 1 + 0.5
            torch.testing.assert_close(masked, expected)
            self.assertEqual(agent._post_update_approx_kl(batch, old, torch.zeros(2)).item(), 0.0)
            self.assertTrue(agent._target_kl_exceeded(torch.tensor(float('nan'))))
            self.assertTrue(agent._target_kl_exceeded(torch.tensor(float('inf'))))
            agent.target_kl = None
            calls = agent.model.a2c_network.calls
            # KL diagnostics remain available even with early stopping disabled.
            torch.testing.assert_close(
                agent._post_update_approx_kl(batch, old, torch.tensor([1.0, 0.0])), masked)
            self.assertEqual(agent.model.a2c_network.calls, calls + 1)
            self.assertFalse(agent._target_kl_exceeded(torch.tensor(1.0)))

    def test_weighted_distributed_reduction(self):
        for repo in REPOS:
            for rank in (0, 1):
                agent = self.agent(repo)
                agent.multi_gpu = True
                # Rank 0: one sample with KL ~0.1065. Rank 1: three unchanged samples.
                agent.model.a2c_network.mean.data.fill_(1.0 if rank == 0 else 0.0)
                sample_count = 1 if rank == 0 else 3
                actions = torch.zeros(sample_count, 1)
                old = -torch.distributions.Normal(0.0, 1.0).log_prob(actions).sum(-1)
                remote_sum = 0.0 if rank == 0 else (torch.exp(torch.tensor(-0.5)) - 0.5).item()
                remote_count = 3.0 if rank == 0 else 1.0
                def average(value, name):
                    return (value + (remote_sum if name == 'target_kl_sum' else remote_count)) / 2
                agent.hvd = SimpleNamespace(average_value=average)
                kl = agent._post_update_approx_kl({'prev_actions': actions}, old)
                expected = (torch.exp(torch.tensor(-0.5)) - 0.5) / 4
                torch.testing.assert_close(kl, expected)
                self.assertTrue(agent._target_kl_exceeded(kl))

    def test_config_validation(self):
        for repo in REPOS:
            agent = self.agent(repo)
            agent._load_config_params({'learning_rate': 2e-5})
            self.assertEqual(agent.target_kl, 0.015)
            for disabled in (0, None):
                agent._load_config_params({'learning_rate': 2e-5, 'target_kl': disabled})
                self.assertIsNone(agent.target_kl)
            for invalid in (-0.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    agent._load_config_params({'learning_rate': 2e-5, 'target_kl': invalid})

    def test_actual_gradient_update_reports_post_step_kl(self):
        for repo in REPOS:
            agent = self.agent(repo)
            calc = load_methods(repo, 'common_agent.py', ['calc_gradients'])['calc_gradients']
            agent.is_rnn = False
            agent.last_lr = 0.3
            agent.e_clip = 0.2
            agent.critic_coef = agent.entropy_coef = agent.bounds_loss_coef = 0.0
            agent.clip_value = False
            agent.set_train = lambda: agent.model.train()
            agent._preproc_obs = lambda obs: obs
            agent.bound_loss = lambda mu: mu.square().sum(-1)
            agent._actor_loss = lambda old, new, adv, clip: {
                'actor_loss': -(old - new).exp() * adv,
                'actor_clipped': torch.zeros_like(adv, dtype=torch.bool)}
            agent._critic_loss = lambda *args: {'critic_loss': torch.zeros(3)}
            agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.3)
            agent.scaler = SimpleNamespace(scale=lambda loss: loss,
                                           step=lambda optimizer: optimizer.step(), update=lambda: None)
            actions = torch.ones(3, 1)
            old = -torch.distributions.Normal(0.0, 1.0).log_prob(actions).sum(-1)
            calc(agent, dict(old_values=torch.zeros(3), old_logp_actions=old,
                             advantages=torch.ones(3), mu=torch.zeros_like(actions),
                             sigma=torch.ones_like(actions), returns=torch.zeros(3),
                             actions=actions, obs=torch.zeros(3, 1)))
            self.assertAlmostEqual(agent.model.a2c_network.mean.item(), 0.3, places=6)
            logratio = torch.distributions.Normal(0.3, 1.0).log_prob(actions).sum(-1) + old
            expected = (logratio.exp() - 1 - logratio).mean()
            torch.testing.assert_close(agent.train_result['approx_kl'], expected)
            self.assertEqual(agent.train_result['kl_early_stop'].item(), 1.0)
            self.assertEqual(agent.train_result['approx_kl_before'].item(), 0.0)
            self.assertIn(('info/ppo_minibatch_approx_kl_before', 0.0, 0), agent.events)
            after = next(value for name, value, step in agent.events
                         if name == 'info/ppo_minibatch_approx_kl_after' and step == 0)
            self.assertAlmostEqual(after, expected.item(), places=6)
            self.assertEqual(agent.flushes, 1)

    def test_minibatch_logging_rank_and_threshold(self):
        for repo in REPOS:
            agent = self.agent(repo)
            agent._ppo_kl_step = 123
            agent._record_ppo_minibatch_kl('before', torch.tensor(0.004))
            agent._record_ppo_minibatch_kl('after', torch.tensor(0.009))
            self.assertEqual(len(agent.events), 4)
            self.assertTrue(all(step == 123 for _, _, step in agent.events))
            self.assertEqual(agent.flushes, 0)
            agent.rank = 1
            agent._record_ppo_minibatch_kl('after', torch.tensor(0.018))
            self.assertEqual(len(agent.events), 4)
            agent.rank = 0
            agent.target_kl = None
            agent._record_ppo_minibatch_kl('after', torch.tensor(0.018))
            self.assertIn(('info/ppo_minibatch_kl_early_stop', 0.0, 123), agent.events)
            self.assertEqual(agent.flushes, 0)

    def test_actual_rollout_loops_stop_and_reset(self):
        for repo in REPOS:
            for filename in ('common_agent.py', 'amp_agent.py'):
                for schedule in ('legacy', 'standard', 'standard_epoch'):
                    for multi_gpu in (False, True):
                        with self.subTest(repo=repo, file=filename, schedule=schedule, ddp=multi_gpu):
                            agent = self.agent(repo)
                            agent.multi_gpu = multi_gpu
                            agent.hvd = SimpleNamespace(average_value=lambda value, name: value)
                            epoch = load_methods(repo, filename, ['train_epoch'])['train_epoch']
                            agent.is_rnn = False
                            agent.has_central_value = False
                            agent.dataset = [None] * 3
                            agent.mini_epochs_num = 4
                            agent.schedule_type = schedule
                            agent.epoch_num = 1
                            agent.last_lr, agent.entropy_coef = 2e-5, 0.0
                            agent.scheduler = SimpleNamespace(update=lambda lr, entropy, *args: (lr, entropy))
                            agent.update_lr = lambda lr: None
                            agent.set_train = lambda: None
                            agent.prepare_dataset = lambda batch: None
                            agent.algo_observer = SimpleNamespace(after_steps=lambda: None)
                            agent._record_train_batch_info = lambda batch, info: None
                            agent.play_steps = lambda: {'played_frames': 3, 'amp_obs': torch.zeros(3, 1)}
                            agent._update_amp_demos = lambda: None
                            agent._amp_obs_demo_buffer = SimpleNamespace(sample=lambda n: {'amp_obs': torch.zeros(n, 1)})
                            agent._amp_replay_buffer = SimpleNamespace(get_total_count=lambda: 0)
                            agent._store_replay_amp_obs = lambda obs: None
                            agent.calls = 0
                            def update(batch):
                                agent.calls += 1
                                kl = torch.tensor([0.004, 0.009, 0.018][min(agent.calls - 1, 2)])
                                return {'kl': kl, 'approx_kl': kl}
                            agent.train_actor_critic = update
                            info = epoch(agent)
                            self.assertEqual(agent.calls, 3)
                            self.assertEqual(len(info['approx_kl']), 3)
                            # A new rollout can update again; the stop flag is local.
                            epoch(agent)
                            self.assertEqual(agent.calls, 4)
                            agent.target_kl = None
                            epoch(agent)
                            self.assertEqual(agent.calls, 16)
                            agent.target_kl = 0.018
                            agent.train_actor_critic = lambda batch: {'kl': torch.tensor(0.018, dtype=torch.float64),
                                                                    'approx_kl': torch.tensor(0.018, dtype=torch.float64)}
                            self.assertEqual(len(epoch(agent)['approx_kl']), 12)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    unittest.main(warnings='ignore')
