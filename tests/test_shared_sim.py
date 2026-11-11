"""Regression coverage for lost resets in the shared GPU simulator facade."""
import ast
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch


def manager():
    source = Path('tokenhsi/env/shared_sim.py')
    tree = ast.parse(source.read_text())
    tree.body = [x for x in tree.body if not (isinstance(x, ast.ImportFrom) and x.module == 'isaacgym')]
    calls = []
    roots = torch.zeros(4, 13)
    dofs = torch.zeros(2, 2)
    gym = SimpleNamespace(
        acquire_actor_root_state_tensor=lambda sim: roots,
        acquire_dof_state_tensor=lambda sim: dofs,
        refresh_actor_root_state_tensor=lambda sim: calls.append(('refresh',)),
        set_actor_root_state_tensor_indexed=lambda sim, state, ids, n: calls.append(('root', state.clone(), ids.clone())),
        set_dof_state_tensor_indexed=lambda sim, state, ids, n: calls.append(('dof', state.clone(), ids.clone())),
        set_dof_position_target_tensor=lambda sim, values: calls.append(('target', values.clone())),
        simulate=lambda sim: calls.append(('simulate',)),
    )
    ns = dict(gymapi=SimpleNamespace(acquire_gym=lambda: gym),
              gymtorch=SimpleNamespace(wrap_tensor=lambda x: x, unwrap_tensor=lambda x: x))
    exec(compile(tree, str(source), 'exec'), ns)
    obj = ns['SharedSim']({'seed': 42}, None, None, 2)
    obj.sim = object()
    obj.parts[0].end = dict(actor=2, dof=1, rigid_body=2, force_sensor=0)
    obj.parts[1].start = dict(obj.parts[0].end)
    obj.parts[1].end = dict(actor=4, dof=2, rigid_body=4, force_sensor=0)
    obj.dof_actors = [0, 2]
    return obj, calls


def test_both_pool_resets_and_controls_are_submitted_once_before_simulate():
    obj, calls = manager()
    ref, kd = obj.parts
    a = torch.zeros(2, 13); a[:, 0] = torch.tensor([5., 6.])
    b = torch.zeros(2, 13); b[0, 0] = 7
    ref.set_actor_root_state_tensor_indexed(obj.sim, a, torch.tensor([0], dtype=torch.int32), 1)
    ref.set_actor_root_state_tensor_indexed(obj.sim, a, torch.tensor([1], dtype=torch.int32), 1)
    kd.set_actor_root_state_tensor_indexed(obj.sim, b, torch.tensor([0], dtype=torch.int32), 1)
    ref.set_dof_state_tensor_indexed(obj.sim, torch.tensor([[1., 2.]]), torch.tensor([0], dtype=torch.int32), 1)
    kd.set_dof_state_tensor_indexed(obj.sim, torch.tensor([[3., 4.]]), torch.tensor([0], dtype=torch.int32), 1)
    ref.set_dof_position_target_tensor(obj.sim, torch.tensor([8.]))
    kd.set_dof_position_target_tensor(obj.sim, torch.tensor([9.]))
    assert calls == []
    ref.simulate(obj.sim)
    assert [x[0] for x in calls] == ['root', 'dof', 'target', 'simulate']
    assert calls[0][2].tolist() == [0, 1, 2]
    assert calls[0][1][:3, 0].tolist() == [5, 6, 7]
    assert calls[1][2].tolist() == [0, 2]
    assert calls[1][1].tolist() == [[1, 2], [3, 4]]
    assert calls[2][1].tolist() == [8, 9]
    ref.simulate(obj.sim)
    assert [x[0] for x in calls].count('root') == 1


def test_refresh_is_shared_once_per_physics_step():
    obj, calls = manager()
    for p in obj.parts:
        p.refresh_actor_root_state_tensor(obj.sim)
    assert calls == [('refresh',)]
    obj.parts[0].simulate(obj.sim)
    for p in obj.parts:
        p.refresh_actor_root_state_tensor(obj.sim)
    assert calls == [('refresh',), ('simulate',), ('refresh',)]


def test_full_state_setter_keeps_other_partition_and_filters_dof_actors():
    obj, calls = manager()
    obj.parts[0].set_actor_root_state_tensor(obj.sim, torch.ones(2, 13))
    obj.parts[1].set_actor_root_state_tensor(obj.sim, torch.full((2, 13), 2.))
    obj.parts[0].set_dof_state_tensor(obj.sim, torch.ones(1, 2))
    obj.parts[1].set_dof_state_tensor(obj.sim, torch.full((1, 2), 2.))
    obj.parts[0].simulate(obj.sim)
    assert calls[0][2].tolist() == [0, 1, 2, 3]
    assert calls[0][1][:, 0].tolist() == [1, 1, 2, 2]
    assert calls[1][2].tolist() == [0, 2]


def test_native_rng_matches_worker_without_advancing_parent_rng():
    obj, _ = manager()
    def draw():
        return (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    parent = obj._rng_state()
    expected_parent = draw()
    obj._restore_rng(parent)
    with obj.native_context():
        first = draw()
    with obj.native_context():
        second = draw()
    assert draw() == expected_parent
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    assert first == draw()
    assert second == draw()
