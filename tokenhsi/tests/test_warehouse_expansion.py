from copy import deepcopy
from types import SimpleNamespace

import isaacgym
import pytest
import torch

from qualitative.warehouse_expansion import configure, command_counts
from qualitative.warehouse_viewer import install_sides
from utils import warehouse_coop
from utils.before_spec import task_objects
from utils.experiments import native_config


@pytest.mark.parametrize('pushers,pairs,independent_at', [
    (2, 1, 0), (2, 2, 0), (3, 2, 0), (4, 1, 0), (4, 2, 0), (4, 2, 1), (2, 1, 2)])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_expanded_layout(pushers, pairs, independent_at, side, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    env = deepcopy(native_config('warehouse_close_no_wall_cpa')['env'])
    configure(env, pushers, pairs, independent_at)
    cfg = env['sensorScene']['warehouse']
    people, n = env['numAgents'], 8
    sizes = torch.tensor([.4, .4, .35])+torch.rand(n, people, 3)*torch.tensor([.2, .2, .15])
    supports = torch.tensor([.7, .7, .4]).expand(n, people, 3)
    counts = torch.full((n,), pushers)
    groups, pre, tasks, roles = warehouse_coop.sample_roles_graph(sizes,
        env['box']['build']['taskSizeRanges'], env['relationGraph']['template_probabilities'], counts, 1., independent_at)
    assert ((tasks == 5).sum(-1) == pushers).all()
    assert ((pre >= 0).sum(-1) == pairs).all()
    leaders = (pre[:, :, None] == torch.arange(people)).any(1)
    independent = (tasks != 5) & (pre < 0) & ~leaders
    assert (independent.sum(-1) == independent_at).all()
    assert (tasks[independent] == 3).all()
    members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    assert (members[independent] == 1).all()
    box = warehouse_coop.box_layout(counts, counts*0+2, counts*0, cfg)
    assert torch.allclose(box['sizes'], torch.tensor([1.8*pushers, 2., 1.5]).expand(n, 3))
    assert torch.allclose(box['opening'], torch.full((n,), 1.8*pushers+.12))
    assert cfg['cooperative']['room_size'][1] == pytest.approx(8.+1.8*(pushers-2))
    monkeypatch.setattr(warehouse_coop, '_candidates', warehouse_coop._candidates)
    install_sides(SimpleNamespace(), warehouse_coop, side, None)
    layout = warehouse_coop.sample_layout(tasks, pre, roles, sizes, supports,
        torch.zeros_like(pre, dtype=torch.bool), counts, counts*0+2, counts*0, cfg, 128)
    valid = layout['valid']
    assert valid.any() if independent_at else valid.sum() >= n*3//4
    pre = pre[valid]
    layout = {key: value[valid] for key, value in layout.items()}
    objects = task_objects(pre, tasks[valid])[:, :people] & (tasks[valid] != 5)
    wall_limit = cfg['cooperative']['room_size'][1]/2-sizes[valid, :, 1]/2-cfg['source_wall_clearance']
    assert (layout['source'][..., 1].abs()[objects] <= wall_limit[objects]+1e-5).all()
    center = (box['door'][valid]+box['landing'][valid])/2
    half = (box['world_size'][valid]+(box['landing'][valid]-box['door'][valid]).abs())/2
    clear = ((layout['source']-center[:, None]).abs() >= sizes[valid, :, :2]/2+half[:, None]+.15-1e-5).any(-1)
    assert clear[objects].all()
    leaders = (pre[:, :, None] == torch.arange(people)).any(1)
    carriers = leaders | independent[valid]
    y = layout['source'][..., 1][carriers]
    assert (y > 0).all() if side == 'left' else (y < 0).all()
    if pushers > 2:
        assert (y.abs() >= .9*(pushers-2)).all()
    for row in range(len(pre)):
        root = layout['root'][row]
        distance = torch.cdist(root, root)
        distance.fill_diagonal_(float('inf'))
        assert (distance >= .85-1e-5).all()
    distance = (layout['source']-layout['goal']).norm(dim=-1)[carriers]
    assert ((distance >= 1.) & (distance <= 5.)).all()


def test_counts_override_positional_defaults():
    result = command_counts(['--num_agents', '4', '--num_objects', '8'], 4, 2)
    assert result == ['--num_agents', '8', '--num_objects', '16', '--group_spawn_over4']
    result = command_counts(['--num_agents', '8', '--num_objects', '16'], 4, 2, 1)
    assert result == ['--num_agents', '9', '--num_objects', '18', '--group_spawn_over4']


def test_invalid_counts():
    for pushers, pairs in [(1, 1), (5, 1), (2, 0)]:
        with pytest.raises(ValueError):
            configure({}, pushers, pairs)
    with pytest.raises(ValueError):
        configure({}, 2, 1, -1)
