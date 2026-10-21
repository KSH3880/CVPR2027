from pathlib import Path

import isaacgym
import pytest
import torch
import yaml

from env.tasks.multi_agent.room_geometry import (
    compute_wall_cpa_collision_penalty, room_walls, sample_room_layout)
from env.tasks.multi_agent.cliff_geometry import terrain_fall
from env.tasks.multi_agent.bridge_geometry import compute_agent_cpa_collision_penalty
from env.tasks.multi_agent.vision_sensor import uniform_fan_grid, fan_height_observation


@pytest.mark.parametrize('velocity,expected', [((1., 0., 0.), .99 ** 60),
    ((-1., 0., 0.), 0.), ((0., 1., 0.), 0.), ((0., 0., 0.), 0.)])
def test_wall_approach_retreat_parallel_and_stop(velocity, expected):
    position = torch.zeros(1, 1, 3)
    velocity = torch.tensor(velocity).view(1, 1, 3)
    start, end = torch.tensor([[[2., -20.]]]), torch.tensor([[[2., 20.]]])
    risk = compute_wall_cpa_collision_penalty(position, velocity, start, end, .7, 1 / 30, .99)
    torch.testing.assert_close(risk, torch.full((1, 1), expected))


def test_close_retreat_has_no_penalty_and_wall_endpoint_is_finite():
    start, end = torch.tensor([[[2., -1.]]]), torch.tensor([[[2., 1.]]])
    position = torch.tensor([[[1.7, 0., 1.], [0., 2., 1.], [0., 1.35, 1.]]])
    velocity = torch.tensor([[[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]])
    risk = compute_wall_cpa_collision_penalty(position, velocity, start, end, .7, 1., 1.)
    torch.testing.assert_close(risk, torch.tensor([[0., 0., .5 * 2 / (4 + .35 ** 2) ** .5]]))


def test_wall_risk_is_agent_cpa_against_stationary_nearest_point():
    start, end = torch.tensor([[[2., -20.]]]), torch.tensor([[[2., 20.]]])
    position = torch.zeros(1, 2, 3)
    velocity = torch.tensor([[[1., 1., 0.], [1., .2, 0.]]])
    actual = compute_wall_cpa_collision_penalty(position, velocity, start, end, .7, 1 / 30, .95)
    expected = torch.stack([compute_agent_cpa_collision_penalty(torch.tensor([[[0., 0., 1.], [2., 0., 1.]]]),
        torch.stack((v, torch.zeros(3))).view(1, 2, 3), .7, 1 / 30, .95)[:, 0] for v in velocity[0]], -1)
    torch.testing.assert_close(actual, expected)
    # An oblique heading that misses the nearest wall point by more than the safety distance is free.
    assert actual[0, 0] == 0 and actual[0, 1] > 0


def test_random_layout_stays_inside_without_overlaps():
    torch.manual_seed(29)
    half = 3 + torch.rand(2048, 2)
    boxes = .25 + .4 * torch.rand(2048, 3, 3)
    humans, objects = sample_room_layout(half, boxes, .8, .2)
    radius = boxes[..., :2].norm(dim=-1) / 2
    assert ((objects.abs() + radius[..., None] + .2) <= half[:, None] + 1e-6).all()
    assert (humans.abs() + .8 <= half[:, None] + 1e-6).all()
    pos = torch.cat((objects, humans), 1)
    radii = torch.cat((radius, torch.full((2048, 2), .4)), 1)
    separation = (pos[:, :, None] - pos[:, None]).norm(dim=-1)
    required = radii[:, :, None] + radii[:, None] + .2
    assert (separation.masked_fill(torch.eye(5, dtype=torch.bool)[None], torch.inf) >= required - 1e-6).all()


def test_sensor_observes_real_wall_heights_across_room_sizes_and_headings():
    n = 128
    half = 3 + torch.rand(n, 2)
    centers, sizes, _, _ = room_walls(half, .4, 2.6)
    quat = torch.zeros(n, 4, 4)
    quat[..., 3] = 1
    root = torch.zeros(n, 2, 3)
    root[..., :2] = (torch.rand(n, 2, 2) * 2 - 1) * (half[:, None] - .8)
    root[..., 2] = 1.
    eye = root.clone()
    eye[..., 2] = 1.9
    grid = uniform_fan_grid(32, 10., 100., 'cpu')[0]
    data = fan_height_observation(eye, root, torch.rand(n, 2) * (2 * torch.pi),
        grid, centers, quat, sizes, torch.zeros(n), 3.)
    hits = data['observation'] > .5
    assert hits.any(-1).all()
    torch.testing.assert_close(data['observation'][hits], torch.full_like(data['observation'][hits], 1.6 / 3))


def test_wall_contact_keeps_episode_alive_but_falling_terminates():
    positions = torch.tensor([[[[3.05, 0., .9], [3.05, 0., .17]]]])
    forces = torch.ones_like(positions)
    ground = torch.full((1, 1, 2), .1)
    args = (torch.tensor([1]), ground, torch.tensor([.15, .15]), torch.tensor([3]), torch.zeros(1))
    assert not terrain_fall(forces, positions, *args).any()
    positions[..., 0, 2] = .2
    assert terrain_fall(forces, positions, *args).all()


def test_room_config_preserves_single_collision_weight_and_sensor_contract():
    base = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    room = yaml.safe_load((base / 'approach_stage2_fan_room.yaml').read_text())['env']
    bridge = yaml.safe_load((base / 'approach_stage2_fan_bridge.yaml').read_text())['env']
    for key in ('agentCollisionCoeff', 'agentCollisionDist', 'agentCollisionTTCDiscount', 'relationReward'):
        assert room[key] == bridge[key]
    assert room['cliff']['sensor_resolution'] == 32 and room['cliff']['sensor_occlusion'] == 'none'
