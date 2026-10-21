from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.cliff_geometry import sensor_grid
from env.tasks.multi_agent.vision_sensor import (
    fan_grid, fan_observation, fill_occluded, ray_box_interval, rotation_matrix)
from env.tasks.multi_agent.vision_sensor_debug import draw_fan_sensor
from learning.multi_agent.cliff_sensor import check_sensor_checkpoint


def scene(grid, centers, sizes, eye=(0., 0., 3.), quaternions=None):
    centers = torch.tensor([centers], dtype=torch.float)
    sizes = torch.tensor([sizes], dtype=torch.float)
    if quaternions is None:
        quaternions = torch.zeros(1, centers.shape[1], 4)
        quaternions[..., 3] = 1
    eye = torch.tensor([[eye]], dtype=torch.float)
    root = eye.clone()
    root[..., 2] -= .6
    return fan_observation(eye, root, torch.zeros(1, 1), grid, centers, quaternions,
                           sizes / 2, torch.zeros(1), int(len(grid)**.5), 3.)


def test_fan_count_range_heading_and_nonuniform_spacing():
    grid = fan_grid(32, 20., 120., .15, 2., 'cpu').view(32, 32, 2)
    radii = grid.norm(dim=-1)
    angles = torch.atan2(grid[..., 1], grid[..., 0])
    assert grid.numel() == 2048
    torch.testing.assert_close(radii[:, -1], torch.full((32,), 20.))
    torch.testing.assert_close(radii[:, 0], torch.full((32,), .15))
    torch.testing.assert_close(angles[[0, -1], 0], torch.tensor([-torch.pi/3, torch.pi/3]))
    assert (radii[:, 1:] > radii[:, :-1]).all()
    assert (radii[:, 1] - radii[:, 0] < radii[:, -1] - radii[:, -2]).all()


def test_fill_uses_only_nearest_farther_visible_on_same_ray():
    surface = torch.tensor([[1.5, 99., 98., 0., 97.], [8., 7., 6., 5., 4.]])
    visible = torch.tensor([[True, False, False, True, False], [False]*5])
    filled, status = fill_occluded(surface, visible)
    torch.testing.assert_close(filled[0, :4], torch.tensor([1.5, 0., 0., 0.]))
    assert status.tolist() == [[1, 2, 2, 1, 0], [0]*5]


def test_cliff_near_ground_occluded_but_far_ground_visible():
    grid = torch.tensor([[r, 0.] for _ in range(4) for r in (1., 3., 6., 7.)])
    result = scene(grid, [[0., 0., .75]], [[4., 20., 1.5]])
    assert result['status'][0, 0, :4].tolist() == [1, 2, 1, 1]
    torch.testing.assert_close(result['surface'][0, 0, :4], torch.tensor([1.5, 0., 0., 0.]))
    torch.testing.assert_close(result['filled'][0, 0, :4], torch.tensor([1.5, 0., 0., 0.]))
    torch.testing.assert_close(result['observation'][0, 0, :4], torch.tensor([-.3, -.8, -.8, -.8]))


def test_tall_box_blocks_far_ground_and_unknown_stays_outside_clip():
    grid = torch.tensor([[r, 0.] for _ in range(4) for r in (1., 3., 6., 20.)])
    result = scene(grid, [[3., 0., 3.]], [[1., 4., 6.]])
    assert result['status'][0, 0, :4].tolist() == [1, 0, 0, 0]
    assert result['observation'][0, 0, 1:4].tolist() == [2., 2., 2.]


def test_visible_box_top_is_retained_and_shadow_uses_ground():
    grid = torch.tensor([[r, 0.] for _ in range(4) for r in (1., 3., 5., 10.)])
    result = scene(grid, [[3., 0., .75]], [[2., 3., 1.5]])
    assert result['status'][0, 0, :4].tolist() == [1, 1, 2, 1]
    torch.testing.assert_close(result['surface'][0, 0, :4], torch.tensor([0., 1.5, 0., 0.]))
    assert result['filled'][0, 0, 2] == 0


def test_thin_occluder_between_samples_is_not_missed():
    grid = torch.tensor([[r, 0.] for _ in range(2) for r in (1., 10.)])
    result = scene(grid, [[2.5, 0., 2.]], [[.05, 4., 4.]])
    assert result['status'][0, 0].tolist() == [1, 0, 1, 0]


def test_head_above_carried_box_can_see_far_ground():
    grid = torch.tensor([[r, 0.] for _ in range(2) for r in (.2, 10.)])
    head = scene(grid, [[.6, 0., 1.]], [[.5, 1., .6]], eye=(0., 0., 1.7))
    pelvis = scene(grid, [[.6, 0., 1.]], [[.5, 1., .6]], eye=(0., 0., 1.))
    assert head['status'][0, 0, 1] == 1
    assert pelvis['status'][0, 0, 1] == 0


def test_rotated_box_vertical_surface_and_parallel_ray():
    angle = torch.tensor(torch.pi / 8)
    q = torch.tensor([[[0., angle.sin(), 0., angle.cos()]]])
    grid = torch.tensor([[r, 0.] for _ in range(2) for r in (2., 4.)])
    result = scene(grid, [[2., 0., 1.]], [[2., 2., 1.]], quaternions=q)
    torch.testing.assert_close(result['surface'][0, 0, 0], torch.tensor(1 + 2**-.5))
    rotation = rotation_matrix(torch.tensor([[0., 0., 0., 1.]]))
    start = torch.tensor([[[[0., 2., 0.], [0., 0., 0.]]]])
    direction = torch.tensor([[[[1., 0., 0.], [1., 0., 0.]]]])
    near, far = ray_box_interval(start, direction, torch.zeros(1, 3), rotation, torch.ones(1, 3))
    assert far[0, 0, 0] < near[0, 0, 0]
    assert near[0, 0, 1] == -1 and far[0, 0, 1] == 1


def test_ray_box_against_independent_scalar_reference():
    import numpy as np
    from scipy.spatial.transform import Rotation

    rng = np.random.RandomState(42)
    centers = rng.randn(3, 3)
    half = rng.uniform(.2, 2., (3, 3))
    rotations = Rotation.random(3, random_state=rng)
    start = rng.randn(3, 2, 11, 3) * 3
    direction = rng.randn(3, 2, 11, 3)
    near, far = ray_box_interval(torch.tensor(start), torch.tensor(direction),
        torch.tensor(centers), rotation_matrix(torch.tensor(rotations.as_quat())), torch.tensor(half))
    for env in range(3):
        for agent in range(2):
            for sample in range(11):
                p = rotations[env].inv().apply(start[env, agent, sample] - centers[env])
                d = rotations[env].inv().apply(direction[env, agent, sample])
                intervals = [sorted(((-half[env, axis] - p[axis]) / d[axis],
                                     (half[env, axis] - p[axis]) / d[axis])) for axis in range(3)]
                np.testing.assert_allclose(near[env, agent, sample], max(row[0] for row in intervals))
                np.testing.assert_allclose(far[env, agent, sample], min(row[1] for row in intervals))


def test_translation_rotation_and_environment_isolation():
    grid = fan_grid(32, 20., 120., .15, 2., 'cpu')
    origins = torch.tensor([[0., 0., 0.], [128., -256., 4.]])
    eye = origins[:, None] + torch.tensor([[[0., 0., 3.]]])
    centers = origins[:, None] + torch.tensor([[[0., 0., .75], [4., 0., 1.]]])
    half = torch.tensor([[[2., 3., .75], [.5, 1., 1.]]]).expand(2, -1, -1)
    q = torch.zeros(2, 2, 4); q[..., 3] = 1
    result = fan_observation(eye, eye, torch.zeros(2, 1), grid, centers, q, half, origins[:, 2], 32, 3.)
    torch.testing.assert_close(result['observation'][0], result['observation'][1], atol=1e-5, rtol=0)
    torch.testing.assert_close(result['status'][0], result['status'][1])
    yaw = torch.full((2, 1), torch.pi/2)
    centers[..., :2] = origins[:, None, :2] + torch.tensor([[[0., 0.], [0., 4.]]])
    q[..., 2] = torch.sin(yaw/2); q[..., 3] = torch.cos(yaw/2)
    rotated = fan_observation(eye, eye, yaw, grid, centers, q, half, origins[:, 2], 32, 3.)
    torch.testing.assert_close(result['observation'], rotated['observation'], atol=2e-5, rtol=0)


def test_same_dimension_does_not_allow_grid_checkpoint():
    grid = {'cliff': {'sensor_resolution': 32, 'sensor_extent': 2.}}
    fan = {'cliff': {'sensor_resolution': 32, 'sensor_pattern': 'fan'}}
    with pytest.raises(ValueError, match='mismatch'):
        check_sensor_checkpoint({'sensor_metadata': grid}, SimpleNamespace(sensor_metadata=fan))


def test_uniform_rings_have_equal_cell_area_and_nearest_outward_links():
    from env.tasks.multi_agent.vision_sensor import uniform_fan_grid
    grid, links = uniform_fan_grid(24, 20., 100., 'cpu')
    radii = grid.norm(dim=-1)
    angles = torch.atan2(grid[:, 1], grid[:, 0])
    assert grid.shape == (576, 2)
    for ring in range(24):
        ids = torch.arange(ring**2, (ring + 1)**2)
        torch.testing.assert_close(radii[ids], torch.full((2 * ring + 1,), (ring + .5) * 20 / 24))
        if ring < 23:
            candidates = torch.arange((ring + 1)**2, (ring + 2)**2)
            nearest = (angles[ids, None] - angles[candidates]).abs().argmin(-1)
            torch.testing.assert_close(links[ids], candidates[nearest])
    assert angles.abs().max() < torch.pi * 50 / 180
    assert [int(((radii >= a) & (radii < a + 5)).sum()) for a in (0, 5, 10, 15)] == [36, 108, 180, 252]


def test_ring_fill_matches_scalar_chain_and_preserves_direct_observations():
    from env.tasks.multi_agent.vision_sensor import uniform_fan_grid, fill_occluded_rings
    _, links = uniform_fan_grid(24, 20., 100., 'cpu')
    generator = torch.Generator().manual_seed(17)
    surface = torch.randn(3, 2, 576, generator=generator)
    visible = torch.rand(3, 2, 576, generator=generator) < .15
    visible[0] = False
    visible[1, :, -47:] = True
    filled, status = fill_occluded_rings(surface, visible, links, 24)
    for env in range(3):
        for agent in range(2):
            for point in range(576):
                source = point
                while not visible[env, agent, source] and links[source] != source:
                    source = links[source].item()
                expected = 1 if visible[env, agent, point] else (2 if visible[env, agent, source] else 0)
                assert status[env, agent, point] == expected
                if expected:
                    assert filled[env, agent, point] == surface[env, agent, source]


def test_grouped_objects_match_serial_with_rotated_boxes_and_partial_group():
    from env.tasks.multi_agent.vision_sensor import uniform_fan_grid
    torch.manual_seed(3)
    grid, links = uniform_fan_grid(24, 20., 100., 'cpu')
    eye = torch.randn(3, 2, 3); eye[..., 2] = 3
    centers = torch.randn(3, 5, 3) * 3
    q = torch.randn(3, 5, 4); q /= q.norm(dim=-1, keepdim=True)
    half = torch.rand(3, 5, 3) + .1
    args = (eye, eye, torch.zeros(3, 2), grid, centers, q, half, torch.zeros(3), 24, 3.)
    reference = fan_observation(*args, successors=links)
    for size in (2, 4):
        result = fan_observation(*args, successors=links, object_batch_size=size)
        for key in ('surface', 'observation', 'status'):
            torch.testing.assert_close(result[key], reference[key], atol=1e-5, rtol=1e-5)


def test_square_viewer_outline_follows_the_grid_edge():
    n = 5
    grid = sensor_grid(n, 10., 'cpu')
    data = {'xy': grid[None], 'filled': torch.zeros(1, n * n), 'eye': torch.zeros(1, 3),
            'status': torch.ones(1, n * n, dtype=torch.long)}
    calls = []
    gym = SimpleNamespace(add_lines=lambda viewer, env, count, lines, colors: calls.append(torch.tensor(lines)))
    draw_fan_sensor(gym, None, None, data, torch.zeros(3), n, square=True)
    outline = calls[1][..., :2]
    assert len(outline) == 4 * (n - 1)
    torch.testing.assert_close(outline[1:, 0], outline[:-1, 1])
    torch.testing.assert_close(outline[0, 0], outline[-1, 1])
    torch.testing.assert_close(outline.abs().amax(-1), torch.full((len(outline), 2), 10.))
    torch.testing.assert_close((outline[:, 1] - outline[:, 0]).norm(dim=-1), torch.full((len(outline),), 5.))
