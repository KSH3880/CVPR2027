from types import SimpleNamespace

import isaacgym
import pytest
import torch

from qualitative.warehouse_viewer import install_sides, install_top_view
from utils import warehouse_coop
from utils.experiments import native_config
from utils.sensor_stage2_spec import WAREHOUSE


@pytest.mark.parametrize('box_side,leader_side', [
    ('left', 'left'), ('left', 'right'), ('right', 'left'), ('right', 'right')])
def test_real_layouts_respect_independent_sides(monkeypatch, box_side, leader_side):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    env = native_config('warehouse_close_random_heading')['env']
    cfg = env['sensorScene']['warehouse']
    n = 32
    sizes = torch.tensor([.4, .4, .35])+torch.rand(n, 4, 3)*torch.tensor([.2, .2, .15])
    supports = torch.tensor([.7, .7, .4]).expand(n, 4, 3)
    counts = torch.full((n,), 2)
    _, pre, tasks, roles = warehouse_coop.sample_roles_graph(sizes,
        env['box']['build']['taskSizeRanges'], env['relationGraph']['template_probabilities'], counts, 1.)
    monkeypatch.setattr(warehouse_coop, '_candidates', warehouse_coop._candidates)
    before = SimpleNamespace()
    install_sides(before, warehouse_coop, box_side, leader_side)
    layout = warehouse_coop.sample_layout(tasks, pre, roles, sizes, supports,
        torch.zeros_like(pre, dtype=torch.bool), counts, counts.clone(), counts*0, cfg, 128)
    valid = layout['valid']
    assert valid.sum() >= n // 4
    leader = pre[valid].max(dim=1).values
    for key, side in (('root', leader_side), ('source', box_side)):
        lateral = layout[key][valid][torch.arange(valid.sum()), leader, 1]
        assert (lateral > 0).all() if side == 'left' else (lateral < 0).all()


def test_variant_override_only_changes_reset_rows():
    before = SimpleNamespace()
    install_sides(before, SimpleNamespace(_candidates=lambda: None), 'random', None)
    task = SimpleNamespace(_warehouse_before_pairs=torch.ones(4, dtype=torch.bool),
                           _warehouse_variant_samples=torch.zeros(2))
    before.sample(task, torch.tensor([1, 3]), torch.tensor([True, True]))
    assert task._warehouse_before_pairs.tolist() == [True, False, True, False]
    assert task._warehouse_variant_samples.tolist() == [2., 0.]


def test_top_camera_tracks_rotated_room_with_environment_offset():
    calls = []
    task_class = type('Task', (), {'_update_camera': lambda self: None})
    install_top_view(task_class)
    task = task_class()
    task.viewer, task.envs = object(), [object()]
    task._structure_yaw = torch.full((1, WAREHOUSE+1), torch.pi/2)
    task._structure_shift = torch.tensor([[[3., 4.]]]).expand(1, WAREHOUSE+1, 2)
    task._env_origins = torch.tensor([[100., 200., 0.]])
    task._warehouse = {'floor_height': .1}
    task.gym = SimpleNamespace(get_env_origin=lambda env: SimpleNamespace(x=10., y=20., z=0.),
                               viewer_camera_look_at=lambda *args: calls.append(args))
    task._update_camera()
    eye, target = calls[0][-2:]
    assert (eye.x, eye.y, eye.z) == pytest.approx((93., 181.99, 18.1))
    assert (target.x, target.y, target.z) == pytest.approx((93., 182., .1))
    task.viewer = None
    task._update_camera()
    assert len(calls) == 1
