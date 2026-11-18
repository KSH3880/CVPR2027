from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.warehouse_before_cache import BeforeLayoutCache, STATE_FIELDS, CONTEXT_FIELDS
from utils import stage2_spec
from utils.sensor_stage2_spec import WAREHOUSE


def make_task():
    n, m = 3, 4
    task = SimpleNamespace(device='cpu', _warehouse_before_pairs=torch.tensor([True, False, True]),
        _templates=torch.tensor([[3, 1, 3, 4]]).repeat(n, 1),
        _before_predecessor=torch.tensor([[-1, 0, -1, 2]]).repeat(n, 1),
        _before_near_pairs=torch.zeros(n, m, dtype=torch.bool),
        _logical_box_order=torch.arange(8).repeat(n, 1),
        _agent_box_assignment=torch.arange(m).repeat(n, 1),
        _surface=torch.ones(n, m, dtype=torch.long), _start_surface=torch.ones(n, m, dtype=torch.long),
        _bridge_count=torch.full((n,), 2), _warehouse_direction=torch.full((n,), 2),
        _box_size=torch.ones(n, 8, 3), _env_origins=torch.zeros(n, 3),
        _structure_yaw=torch.zeros(n, 4), _structure_shift=torch.zeros(n, 4, 2),
        _surface_half=torch.ones(n, 4, 2, 2), _place_failed=torch.zeros(n, dtype=torch.bool))
    for index, name in enumerate((*STATE_FIELDS, '_hist_amp_obs_buf')):
        setattr(task, name, torch.arange(n*8).view(n, 4, 2).float()+index*100)
    task._groups = torch.tensor([[0, 0, 2, 2]]).repeat(n, 1)
    task._members, task._regions = torch.ones(n, m, dtype=torch.long), torch.zeros(n, m, dtype=torch.long)
    task._roles = torch.arange(m).repeat(n, 1)
    task._stage2_task_slots = torch.ones(n, 8, dtype=torch.bool)
    task._scene_domains = torch.zeros(n, dtype=torch.long)
    task._unreachable = torch.zeros(n, m, dtype=torch.bool)
    task.relation_runtime = SimpleNamespace(
        graph=stage2_spec.compose_graph(task._templates, task._groups, task._before_predecessor, 8, shuffle=False),
        before=stage2_spec.before_members(task._before_predecessor))
    return task


def test_restore_keeps_the_complete_start_state_and_amp_history_for_only_requested_apron_envs():
    task = make_task()
    cache = BeforeLayoutCache(3, 42, refresh_probability=0., shuffle_edges=False)
    ids = torch.arange(3)
    assert cache.restore(task, ids).numel() == 0
    cache.finish(task, ids)
    expected = {name: getattr(task, name).clone() for name in (*STATE_FIELDS, '_hist_amp_obs_buf')}
    context = {name: getattr(task, name).clone() for name in CONTEXT_FIELDS}
    graph_owner = task.relation_runtime.graph.edge_owner.clone()
    task._logical_box_order.fill_(-1)
    task.relation_runtime.graph.edge_owner.fill_(-1)
    for name in expected:
        getattr(task, name).fill_(-1)
    hits = cache.restore(task, torch.tensor([0, 1]))
    assert hits.tolist() == [0]
    for name in STATE_FIELDS:
        torch.testing.assert_close(getattr(task, name)[0], expected[name][0])
        assert (getattr(task, name)[1:] == -1).all()
    for name in CONTEXT_FIELDS:
        torch.testing.assert_close(getattr(task, name)[0], context[name][0])
    torch.testing.assert_close(task.relation_runtime.graph.edge_owner[0], graph_owner[0])
    assert (task._hist_amp_obs_buf == -1).all()
    cache.finish(task, torch.tensor([1]))
    torch.testing.assert_close(task._hist_amp_obs_buf[0], expected['_hist_amp_obs_buf'][0])
    assert not cache.entries[1]


@pytest.mark.parametrize('name', ['_templates', '_before_predecessor', '_before_near_pairs',
    '_box_size', '_env_origins', '_bridge_count', '_warehouse_direction',
    '_structure_yaw', '_structure_shift', '_surface_half'])
def test_different_task_conditions_or_geometry_do_not_reuse_a_cached_layout(name):
    task = make_task()
    cache = BeforeLayoutCache(3, 42, refresh_probability=0.)
    ids = torch.tensor([0])
    cache.restore(task, ids)
    cache.finish(task, ids)
    values = getattr(task, name)[0]
    if name.startswith('_structure') or name == '_surface_half':
        values = values[WAREHOUSE]
    values = values.reshape(-1)
    index = 1 if name in ('_templates', '_before_predecessor', '_before_near_pairs') else 0
    if name == '_before_predecessor':
        values[index] = -1
    else:
        values[index] = ~values[index] if values.dtype == torch.bool else values[index]+1
    assert cache.restore(task, ids).numel() == 0


def test_pool_keeps_multiple_independent_snapshots_and_bounds_conditions():
    task = make_task()
    cache = BeforeLayoutCache(3, 42, capacity=2, samples=2, refresh_probability=1.)
    ids = torch.tensor([0])
    for value in (1., 2., 3.):
        cache.restore(task, ids)
        task._humanoid_root_states[0].fill_(value)
        cache.finish(task, ids)
    pool = next(iter(cache.entries[0].values()))
    assert len(pool) == 2
    assert [state['_humanoid_root_states'][0, 0].item() for state in pool] == [2., 3.]
    for value in (1, 3):
        task._before_near_pairs[0, value] = True
        cache.restore(task, ids)
        cache.finish(task, ids)
    assert len(cache.entries[0]) == 2
    assert cache.refreshes == 2


def test_same_team_conditions_restore_the_cached_owner_bindings_and_graph():
    task = make_task()
    cache = BeforeLayoutCache(3, 42, refresh_probability=0., shuffle_edges=False)
    ids = torch.tensor([0])
    cache.restore(task, ids)
    cache.finish(task, ids)
    expected = task._templates[0].clone()
    owners = task.relation_runtime.graph.edge_owner[0].clone()
    task._templates[0] = torch.tensor([3, 4, 3, 1])
    task._logical_box_order[0] = torch.arange(8).flip(0)
    task.relation_runtime.graph.edge_owner[0] = -1
    assert cache.restore(task, ids).tolist() == [0]
    torch.testing.assert_close(task._templates[0], expected)
    torch.testing.assert_close(task.relation_runtime.graph.edge_owner[0], owners)
    torch.testing.assert_close(task._logical_box_order[0], torch.arange(8))


def test_empty_requests_and_diagnostics():
    task = make_task()
    cache = BeforeLayoutCache(3, 42, refresh_probability=0.)
    assert cache.restore(task, torch.tensor([], dtype=torch.long)).numel() == 0
    cache.restore(task, torch.tensor([0]))
    cache.finish(task, torch.tensor([0]))
    cache.restore(task, torch.tensor([0]))
    cache.finish(task, torch.tensor([], dtype=torch.long))
    result = {}
    cache.diagnostics(task, result)
    assert result['warehouse/before_cache/requests'] == 2
    assert result['warehouse/before_cache/hits'] == 1
    assert result['warehouse/before_cache/hit_fraction'] == .5
    assert result['warehouse/before_cache/entries'] == 1
    assert cache.requests == cache.hits == cache.refreshes == 0


def test_warmup_populates_once_without_changing_variant_counters_or_random_streams():
    task = make_task()
    task._scene_ids = torch.full((3,), WAREHOUSE)
    names = ('_sampling_counts', '_sampling_retries', '_scene_sampling',
             '_stage2_domain_counts', '_size_rsi_executed', '_relation_episode_id')
    for name in names:
        setattr(task, name, torch.tensor([4]))
    cache = BeforeLayoutCache(3, 42, warmup_resets=3)
    ids = torch.arange(3)
    flags = task._warehouse_before_pairs.clone()
    rng, cache_rng = torch.get_rng_state(), cache.random.getstate()
    calls = []

    def reset(selected):
        cache.warmup(task, selected)
        calls.append(selected)
        for name in names:
            getattr(task, name).add_(1)
        torch.rand(5)
        cache.random.random()
        assert task._warehouse_before_pairs.all()
        assert not len(cache.restore(task, selected))
        cache.finish(task, selected)

    task._reset_ontop_context_envs = reset
    cache.warmup(task, ids)
    cache.warmup(task, ids)
    assert len(calls) == 3
    assert all(len(next(iter(entries.values()))) == 2 for entries in cache.entries)
    torch.testing.assert_close(task._warehouse_before_pairs, flags)
    for name in names:
        torch.testing.assert_close(getattr(task, name), torch.tensor([4]))
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert cache.random.getstate() == cache_rng
    assert cache.refresh_probability == .25 and cache.requests == cache.hits == cache.refreshes == 0
