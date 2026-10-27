from pathlib import Path

import torch
import yaml
import pytest

from env.tasks.multi_agent.bridge_push4_scene import task_layout
from utils.bridge_push4_spec import episode_graph, sample_roles
from utils.bridge_spec import predecessor_relations, validate_env


def test_roles_tasks_and_directed_dependencies():
    torch.manual_seed(42)
    roles, tasks, surface, start = sample_roles(8192, 'cpu')
    rows = torch.arange(len(tasks))
    graph, predecessor = episode_graph(tasks, roles, 8)
    assert (tasks[rows, roles[:, 0]] == 4).all()
    assert (start[rows, roles[:, 0]] == 0).all()
    assert (start[rows, roles[:, 1]] == 1).all()
    assert (surface[rows, roles[:, 1]] == 2).all()
    assert (predecessor >= 0).sum(-1).eq(1).all()
    assert torch.equal(predecessor[rows, roles[:, 1]], roles[:, 0])
    relations = predecessor_relations(predecessor, 4)
    assert (relations == 1).sum(-1).eq(1).all()
    assert (relations == 2).sum((1, 2)).eq(1).all()
    assert graph.prereq_mask.sum((1, 2)).eq(1).all()
    independent = tasks.gather(1, roles[:, 2:]).flatten()-1
    assert (torch.bincount(independent)/len(independent)-.25).abs().max() < .02


@pytest.mark.parametrize('extreme', [False, True])
def test_safe_random_geometry_and_non_aligned_crossings(extreme):
    torch.manual_seed(23)
    count = 128 if extreme else 1024
    roles, tasks, surface, start = sample_roles(count, 'cpu')
    side = torch.empty(count, 3).uniform_(1.2, 1.7)
    side[:] = side[:, :1]
    if extreme:
        side[:] = 1.7
    sizes = torch.full((count, 4, 3), .8 if extreme else .5)
    sizes[tasks == 4] = side[:, None].expand(-1, 4, -1)[tasks == 4]
    half_a = torch.randint(8, 17, (count, 2))*.25
    half_b = torch.randint(8, 17, (count, 2))*.25
    if extreme:
        half_a[:], half_b[:] = 2., 2.
    cfg = dict(translation_range=[-2., 2.], landing_margin=.8, approach_distance=[.3, 1.],
               edge_margin=.5, human_margin=.8, clearance=.05)
    layout = task_layout(roles, tasks, surface, start, sizes, half_a, half_b, side, cfg)
    torch.testing.assert_close(layout['far'][:, 0]-half_b[:, 0]-(layout['near'][:, 0]+half_a[:, 0]), side[:, 0]+.1)
    for key, half in (('near', half_a), ('far', half_b)):
        distance = (layout['landing'][:, 1]-layout[key][:, 1]).abs()+side[:, 1]/2
        assert (distance <= half[:, 1]-.8+1e-5).all()
    rows = torch.arange(count)
    human = layout['humans'][rows, roles[:, 1]]
    box = layout['boxes'][rows, roles[:, 1]]
    fraction = (layout['landing'][:, 0]-human[:, 0])/(box[:, 0]-human[:, 0])
    line_y = human[:, 1]+fraction*(box[:, 1]-human[:, 1])
    aligned = (line_y-layout['landing'][:, 1]).abs() < side[:, 1]/2
    assert .05 < aligned.float().mean() < .95


def test_new_contract():
    root = Path(__file__).resolve().parents[2]
    env = yaml.safe_load((root/'tokenhsi/data/cfg/multi_agent/stage2_bridge_push4.yaml').read_text())['env']
    validate_env(env)


def test_mixed_start_configuration_changes_only_sampling():
    root = Path(__file__).resolve().parents[2]
    cfg = root/'tokenhsi/data/cfg/multi_agent'
    original = yaml.safe_load((cfg/'stage2_bridge_push4.yaml').read_text())
    mixed = yaml.safe_load((cfg/'stage2_bridge_push4_mix50.yaml').read_text())
    validate_env(mixed['env'])
    assert mixed['env']['bridge'].pop('start_mixture') == {
        'near_probability': .5, 'near_aligned_probability': .5, 'near_distance': [.1, 1.5]}
    mixed['experiment']['name'] = original['experiment']['name']
    assert mixed == original


@pytest.mark.parametrize('extreme', [False, True])
def test_mixed_start_distances_and_push_direction(extreme):
    from env.tasks.multi_agent.bridge_push4_scene import stage_layout
    torch.manual_seed(42)
    n = 4096
    side = torch.full((n, 3), 1.7) if extreme else torch.empty(n, 1).uniform_(1.2, 1.7).expand(-1, 3)
    half = torch.full((n, 2), 2.) if extreme else torch.randint(8, 17, (n, 2))*.25
    near = torch.rand(n) < .5
    cfg = dict(translation_range=[-2., 2.], landing_margin=.8, clearance=.05,
        approach_distance=[.3, 1.], start_mixture=dict(near_probability=.5, near_distance=[.1, 1.5]))
    a, b, goal, start, sign = stage_layout(half, half, side, cfg, near)
    distance = (start-goal).norm(dim=-1)
    assert .47 < near.float().mean() < .53
    assert ((distance[near] >= .1-1e-6) & (distance[near] <= 1.5+1e-6)).all()
    assert (distance[~near] > 2.).all()
    assert ((start[:, 1]-goal[:, 1])*sign > 0).all()
    torch.testing.assert_close(start[:, 0], goal[:, 0])


def test_layout_retries_preserve_start_mode(monkeypatch):
    from env.tasks.multi_agent import bridge_push4_scene as scene
    torch.manual_seed(42)
    n = 64
    roles, tasks, surface, start = sample_roles(n, 'cpu')
    side = torch.full((n, 3), 1.7)
    sizes = torch.full((n, 4, 3), .5)
    sizes[tasks == 4] = side[:, None].expand(-1, 4, -1)[tasks == 4]
    half = torch.full((n, 2), 2.)
    near = torch.arange(n) % 2 == 0
    aligned = torch.arange(n) % 4 == 0
    cfg = dict(translation_range=[-2., 2.], landing_margin=.8, approach_distance=[.3, 1.],
        edge_margin=.5, human_margin=.8, clearance=.05,
        start_mixture=dict(near_probability=.5, near_distance=[.1, 1.5]))
    original, calls = scene._task_layout, []

    def retry(*args):
        calls.append(torch.stack((args[-2], args[-1])))
        result, failed = original(*args)
        if len(calls) == 1:
            failed[:] = True
        return result, failed

    monkeypatch.setattr(scene, '_task_layout', retry)
    layout = scene.task_layout(roles, tasks, surface, start, sizes, half, half, side, cfg, near, aligned)
    assert len(calls) >= 2
    assert torch.equal(calls[0], calls[1])
    rows = torch.arange(n)
    distance = (layout['boxes'][rows, roles[:, 0]]-layout['landing']).norm(dim=-1)
    assert ((distance[near] >= .1-1e-5) & (distance[near] <= 1.5+1e-5)).all()
    assert (distance[~near] > 2.).all()


def test_near_aligned_layout_keeps_interior_margins_and_random_remainder():
    torch.manual_seed(42)
    n = 512
    roles, tasks, surface, start = sample_roles(n, 'cpu')
    side = torch.empty(n, 1).uniform_(1.2, 1.7).expand(-1, 3)
    sizes = torch.full((n, 4, 3), .5)
    sizes[tasks == 4] = side[:, None].expand(-1, 4, -1)[tasks == 4]
    a, b = torch.randint(8, 17, (n, 2))*.25, torch.randint(8, 17, (n, 2))*.25
    near = torch.rand(n) < .5
    aligned = near & (torch.rand(n) < .5)
    cfg = dict(translation_range=[-2., 2.], landing_margin=.8, approach_distance=[.3, 1.],
        edge_margin=.5, human_margin=.8, clearance=.05,
        start_mixture=dict(near_probability=.5, near_aligned_probability=.5, near_distance=[.1, 1.5]))
    layout = task_layout(roles, tasks, surface, start, sizes, a, b, side, cfg, near, aligned)
    rows = torch.arange(n)
    xy = layout['humans'][rows, roles[:, 1]]
    box = layout['boxes'][rows, roles[:, 1]]
    direction = box-xy
    delta = layout['landing']-xy
    error = (direction[:, 0]*delta[:, 1]-direction[:, 1]*delta[:, 0]).abs()/direction.norm(dim=-1)
    assert error[aligned].max() < 1e-5
    slope = direction[:, 1]/direction[:, 0]
    bridge_margin = side[:, 1]/2-slope.abs()*(side[:, 0]/2+cfg['clearance'])
    assert (bridge_margin[aligned] >= .2-1e-5).all()
    assert (error[~aligned] > .05).float().mean() > .8
    assert ((xy[aligned]-layout['near'][aligned]).abs() <= a[aligned]-.8+1e-5).all()
    assert ((box-layout['far']).abs()+sizes[rows, roles[:, 1], :2]/2 <= b-.5+1e-5).all()
    assert .2 < aligned.float().mean() < .3
