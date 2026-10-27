from types import SimpleNamespace

import pytest
import torch
from torch import nn

from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.stage1_push_agent import Stage1PushAgent


@pytest.mark.parametrize('mode', ['resume', 'warm_start', 'fixed'])
def test_zero_kl_skips_teacher_and_loads_only_student_weights(mode, tmp_path, monkeypatch):
    spec = {'teacher_kl_coef':0., 'source_checkpoint':'fixed.pth'}
    if mode == 'warm_start':
        spec['warm_start_checkpoint'] = 'warm.pth'
    task = SimpleNamespace(cfg={'env':{'stage1Push':spec}, 'args':SimpleNamespace(resume=int(mode == 'resume'))})
    loaded = []

    def initialize(agent, *args):
        agent.vec_env = SimpleNamespace(env=SimpleNamespace(task=task))
        agent.ppo_device = 'cpu'
        agent.model = object()
        agent.running_mean_std = nn.Identity()
        agent._amp_input_mean_std = nn.Identity()
        agent.normalize_value = False
        agent.experiment_dir = tmp_path

    def load(path, **kwargs):
        loaded.append(path)
        if path == 'fixed.pth' and mode == 'fixed':
            return {'running_mean_std':{}}
        if path == 'warm.pth' and mode == 'warm_start':
            return {'epoch':7}
        pytest.fail('Unexpected teacher checkpoint load')

    def copy_teacher(value):
        pytest.fail('Teacher model should not be copied')

    monkeypatch.setattr(MAAgent, '__init__', initialize)
    monkeypatch.setattr(MAAgent, 'set_weights', lambda agent, weights: None)
    monkeypatch.setattr('learning.multi_agent.stage1_push_agent.torch.load', load)
    monkeypatch.setattr('learning.multi_agent.stage1_push_agent.deepcopy', copy_teacher)
    monkeypatch.setattr('learning.multi_agent.stage1_push_agent.transfer', lambda model, source: ({},{}))
    monkeypatch.setattr('learning.multi_agent.stage1_push_agent.check_navigation_transfer', lambda weights, env: None)
    agent = Stage1PushAgent('test', {})
    assert agent.teacher is None
    assert loaded == {'resume':[], 'warm_start':['warm.pth'], 'fixed':['fixed.pth']}[mode]


def test_zero_kl_returns_before_reading_observations_or_teacher():
    agent = SimpleNamespace(_push_spec={'teacher_kl_coef':0.})
    mu = torch.ones(4, 3, requires_grad=True)
    loss = Stage1PushAgent._actor_preservation_loss(agent, None, mu, None)
    assert loss.item() == 0 and loss.shape == ()
    (mu.square().sum()+loss).backward()
    torch.testing.assert_close(mu.grad, torch.full_like(mu, 2.))


def test_positive_kl_still_creates_a_frozen_teacher(monkeypatch):
    spec = {'teacher_kl_coef':.001, 'source_checkpoint':'fixed.pth'}
    task = SimpleNamespace(cfg={'env':{'stage1Push':spec}, 'args':SimpleNamespace(resume=1)})

    def initialize(agent, *args):
        agent.vec_env = SimpleNamespace(env=SimpleNamespace(task=task))
        agent.ppo_device = 'cpu'
        agent.model = SimpleNamespace(a2c_network=nn.Linear(2,2))
        agent.running_mean_std = nn.Identity()
        agent._amp_input_mean_std = nn.Identity()

    weights = {'model':{'a2c_network.weight':torch.ones(2,2), 'a2c_network.bias':torch.zeros(2)}}
    monkeypatch.setattr(MAAgent, '__init__', initialize)
    monkeypatch.setattr('learning.multi_agent.stage1_push_agent.torch.load', lambda *args, **kwargs: weights)
    agent = Stage1PushAgent('test', {})
    assert agent.teacher is not None and not agent.teacher.training
    assert all(not p.requires_grad for p in agent.teacher.parameters())
    torch.testing.assert_close(agent.teacher.weight, torch.ones(2,2))
