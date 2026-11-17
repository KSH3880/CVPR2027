from types import SimpleNamespace

import torch
import pytest

from env.tasks.multi_agent.humanoid_ma_sensor_bridge_before import HumanoidMASensorBridgeBefore
from env.tasks.multi_agent.humanoid_ma_sensor_compact import HumanoidMASensorCompact
from utils.bridge_before_fixed import bridge_center_progress
from utils.episode_layout_cache import EpisodeLayoutCache, buckets
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
from env.tasks.multi_agent import humanoid_ma_stage2 as stage2_module


def test_center_progress_approach_and_sticky_saturation_in_both_directions():
    direction = torch.tensor([[1., 0.], [-1., 0.]])
    bridge = torch.tensor([[5., 3.], [-5., -3.]])
    passed = torch.zeros(2, dtype=torch.bool)
    crossing = torch.ones_like(passed)
    previous = torch.zeros(2)
    for distance in (4., 2., .5):
        reward, passed = bridge_center_progress(bridge-distance*direction, bridge, direction, passed, crossing, 1.)
        assert (reward > previous).all() and not passed.any()
        previous = reward
    reward, passed = bridge_center_progress(bridge+.1*direction, bridge, direction, passed, crossing, 1.)
    torch.testing.assert_close(reward, torch.ones(2))
    reward, passed = bridge_center_progress(bridge-20*direction, bridge, direction, passed, crossing, 1.)
    assert passed.all()
    torch.testing.assert_close(reward, torch.ones(2))


def test_same_platform_and_nonleaders_never_receive_center_reward():
    task = SimpleNamespace(num_envs=2, num_agents=4, device='cpu',
        _scene_cfg=dict(bridge=dict(before_training=dict(bridge_center_progress=dict(weight=.2, sigma=1.)))),
        _roles=torch.tensor([[2, 0, 3, 1], [0, 2, 1, 3]]),
        _agent_box_assignment=torch.tensor([[1, 0, 2, 3], [0, 1, 2, 3]]),
        _bridge_slot=torch.tensor([4, 4]), _box_states=torch.zeros(2, 5, 13),
        _bridge_center_direction=torch.tensor([[1., 0.], [1., 0.]]),
        _bridge_center_passed=torch.tensor([True, False]), _before_same_platform=torch.tensor([True, False]))
    task._box_states[1, 2, 0] = -1.
    reward = HumanoidMASensorBridgeBefore._compute_bridge_center_progress(task)
    expected = torch.zeros(2, 4)
    expected[1, 2] = .1
    torch.testing.assert_close(reward, expected)
    assert not task._bridge_center_passed.any()


def test_partial_reset_clears_only_selected_saturation(monkeypatch):
    monkeypatch.setattr(HumanoidMASensorCompact, '_reset_relation_history', lambda self, ids: None)
    task = HumanoidMASensorBridgeBefore.__new__(HumanoidMASensorBridgeBefore)
    task.device = 'cpu'
    task._bridge_center_passed = torch.ones(3, dtype=torch.bool)
    task._bridge_center_direction = torch.zeros(3, 2)
    task._roles = torch.tensor([[0, 1, 2, 3]]).repeat(3, 1)
    task._start_surface = torch.ones(3, 4, dtype=torch.long)
    task._surface_center = torch.zeros(3, 3, 2, 2)
    task._surface_center[:, 1, 1, 0] = 10.
    task._structure_yaw = torch.zeros(3, 3)
    task._before_same_platform = torch.tensor([False, True, False])
    task._before_mix_counts = torch.zeros(2)
    task._reset_relation_history(torch.tensor([1]))
    assert task._bridge_center_passed.tolist() == [True, False, True]


def test_layout_cache_separates_same_platform_and_crossing():
    bank = EpisodeLayoutCache(2, 2, bucket_count=54)
    ids, mode, family = torch.arange(2), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)
    alpha = torch.zeros(2)
    crossing = buckets(mode, family, alpha, torch.zeros(2, dtype=torch.bool))
    local = buckets(mode, family, alpha, torch.ones(2, dtype=torch.bool))
    assert (local-crossing).eq(27).all()
    state = dict(same=torch.zeros(2, dtype=torch.bool))
    bank.store(crossing, ids, alpha, state)
    assert bank.lookup(local, ids, alpha).eq(-1).all()
    state['same'][:] = True
    bank.store(local, ids, alpha, state)
    bank.restore(crossing, ids, bank.lookup(crossing, ids, alpha), state)
    assert not state['same'].any()


def test_default_amp_reset_covers_all_reset_slots_after_layout_retries(monkeypatch):
    seen = []
    monkeypatch.setattr(HumanoidMASensorCompact, '_init_amp_obs',
        lambda self, ids: seen.append(self._reset_default_slots))
    task = HumanoidMASensorBridgeBefore.__new__(HumanoidMASensorBridgeBefore)
    task.device, task.num_agents = 'cpu', 4
    task._state_init = task.StateInit.Default
    task._reset_default_slots = (torch.tensor([3]), torch.tensor([1]))
    task._init_amp_obs(torch.tensor([1, 3]))
    assert seen[0][0].tolist() == [1]*4+[3]*4
    assert seen[0][1].tolist() == list(range(4))*2


@pytest.mark.parametrize('method,owners', [('_reset_boxes', [0, 2, 3]), ('_configure_ontop_targets', [0, 3])])
def test_default_placement_uses_same_before_filter_as_loco(monkeypatch, method, owners):
    seen = []
    def parent(self, ids):
        seen.append((self._reset_default_slots[1].tolist(), self._reset_ref_slots['loco'][1].tolist()))
    monkeypatch.setattr(HumanoidMACoupledPush, method, parent)
    monkeypatch.setattr(stage2_module, 'apply_before_layout', lambda self, ids: None)
    task = HumanoidMAStage2.__new__(HumanoidMAStage2)
    task.num_agents = 4
    task._before_predecessor = torch.tensor([[-1, 0, 0, -1]])
    task._templates = torch.tensor([[3, 1, 4, 5]])
    slots = (torch.zeros(4, dtype=torch.long), torch.arange(4))
    task._reset_default_slots = slots
    task._reset_ref_slots = dict(loco=slots)
    getattr(task, method)(torch.tensor([0]))
    assert seen == [(owners, owners)]
    assert task._reset_default_slots is slots and task._reset_ref_slots['loco'] is slots
