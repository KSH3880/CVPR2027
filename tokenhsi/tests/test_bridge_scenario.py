from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from env.tasks.multi_agent.bridge_geometry import (
    bridge_control, bridge_schedule, bridge_trajectory, sample_bridge_layout)
from env.tasks.multi_agent.vision_sensor import (
    fan_height_observation, fan_observation, uniform_fan_grid, vertical_height)
from learning.multi_agent.cliff_sensor import check_sensor_checkpoint

CONFIG = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent/approach_stage2_fan_bridge.yaml'
ENV = yaml.safe_load(CONFIG.read_text())['env']
BRIDGE, CLIFF = ENV['bridge'], ENV['cliff']
RINGS = CLIFF['sensor_resolution']
GRID, SUCCESSORS = uniform_fan_grid(RINGS, CLIFF['sensor_range'], CLIFF['sensor_fov'], 'cpu')


def yaw_quaternion(yaw):
    zero = torch.zeros_like(yaw)
    return torch.stack((zero, zero, (yaw / 2).sin(), (yaw / 2).cos()), -1)


def two_stages(gap, bridge):
    boxes = [((-2., 0., .75), (2., 3., .75)), ((gap + 2., 0., .75), (2., 3., .75))]
    if bridge:
        boxes.append(((gap / 2, 0., .745), (gap / 2 - .01, .75, .745)))
    return (torch.tensor([[b[0] for b in boxes]]), yaw_quaternion(torch.zeros(1, len(boxes))),
            torch.tensor([[b[1] for b in boxes]]))


def test_vertical_height_matches_slab_surface_for_upright_boxes():
    torch.manual_seed(0)
    centers = torch.cat((torch.tensor([[[0., 0., .75]]]).expand(8, 1, 3),
                         torch.cat((torch.rand(8, 4, 2) * 10 - 3, torch.full((8, 4, 1), 1.8)), -1)), 1)
    half = torch.cat((torch.tensor([[[3., 2.5, .75]]]).expand(8, 1, 3),
                      torch.rand(8, 4, 3) * .4 + .1), 1)
    quat = yaw_quaternion(torch.cat((torch.zeros(8, 1), torch.rand(8, 4) * 6.28), 1))
    eye = torch.cat((torch.rand(8, 2, 2) * 2 - 1, torch.full((8, 2, 1), 3.1)), -1)
    heading = torch.rand(8, 2) * 6.28
    reference = fan_observation(eye, eye, heading, GRID, centers, quat, half, torch.zeros(8),
                                RINGS, 3., successors=SUCCESSORS)
    current = fan_height_observation(eye, eye, heading, GRID, centers, quat, half, torch.zeros(8), 3.)
    torch.testing.assert_close(current['surface'], reference['surface'], atol=1e-4, rtol=0)
    assert (current['status'] == 1).all()


def test_vertical_height_uses_highest_corner_of_tilted_box():
    angle = torch.tensor(.3)
    quat = torch.tensor([[[(angle / 2).sin(), 0., 0., (angle / 2).cos()]]])
    half = torch.tensor([[[.5, .4, .2]]])
    height = vertical_height(torch.zeros(1, 1, 2), torch.tensor([[[0., 0., 1.]]]), quat, half, torch.zeros(1))
    expected = 1 + .4 * angle.sin() + .2 * angle.cos()
    torch.testing.assert_close(height[0, 0], expected)


@pytest.mark.parametrize('gap', [1.5, 2.5])
def test_gap_and_bridge_are_distinguishable_from_far_away(gap):
    eye, root = torch.tensor([[[-5., 0., 3.1]]]), torch.tensor([[[-5., 0., 2.4]]])

    def observe(bridge):
        return fan_height_observation(eye, root, torch.zeros(1, 1), GRID, *two_stages(gap, bridge),
                                      torch.zeros(1), 3.)

    empty, bridged = observe(False), observe(True)
    xy = empty['xy'][0, 0]
    channel = (xy[:, 0] > .02) & (xy[:, 0] < gap - .02)
    band = channel & (xy[:, 1].abs() < .7)
    assert band.sum() > 5
    assert (empty['surface'][0, 0][channel] == 0).all()
    torch.testing.assert_close(bridged['surface'][0, 0][band], torch.full((int(band.sum()),), 1.49))
    assert (bridged['surface'][0, 0][channel & (xy[:, 1].abs() > .8)] == 0).all()


def test_fixed_far_stage_keeps_random_bridge_and_task_starts():
    n = 64
    half_a, half_b = torch.full((n, 2), 3.), torch.full((n, 2), 2.5)
    offset = torch.linspace(-2., 2., n)
    gap = torch.full((n,), 2.)
    bridge_half = torch.tensor([[.99, .6]]).expand(n, -1)
    boxes = torch.full((n, 3, 3), .4)
    first = sample_bridge_layout(half_a, half_b, gap, bridge_half, boxes, BRIDGE, .8, .5, far_offset=offset)
    second = sample_bridge_layout(half_a, half_b, gap, bridge_half, boxes, BRIDGE, .8, .5, far_offset=offset)
    torch.testing.assert_close(first['far_stage'], second['far_stage'])
    torch.testing.assert_close(first['far_stage'][:, 1], offset)
    for key in ('humans', 'objects', 'landing', 'start'):
        assert not torch.equal(first[key], second[key])
    for layout in (first, second):
        radius = boxes[..., :2].norm(dim=-1) / 2
        assert ((layout['objects'] - layout['far_stage'][:, None]).abs() + radius[..., None] <= half_b[:, None]).all()


def test_bridge_layout_invariants():
    torch.manual_seed(1)
    n = 4096
    sizes_a = torch.arange(CLIFF['size_range'][0], CLIFF['size_range'][1] + .01, CLIFF['size_step'])
    sizes_b = torch.arange(BRIDGE['far_size_range'][0], BRIDGE['far_size_range'][1] + .01, CLIFF['size_step'])
    half_a = sizes_a[torch.randint(len(sizes_a), (n, 2))] / 2
    half_b = sizes_b[torch.randint(len(sizes_b), (n, 2))] / 2
    gap = torch.round((1.5 + torch.rand(n)) * 10) / 10
    bridge_half = torch.stack((gap / 2 - .01, torch.round((1 + torch.rand(n)) * 10) / 20), -1)
    boxes = torch.rand(n, 3, 3) * torch.tensor([.25, .25, .3]) + torch.tensor([.4, .4, .25])
    layout = sample_bridge_layout(half_a, half_b, gap, bridge_half, boxes, BRIDGE, .8, .5)
    eps = 1e-5

    far = layout['far_stage']
    torch.testing.assert_close(far[:, 0], half_a[:, 0] + gap + half_b[:, 0])
    overlap_low = torch.maximum(-half_a[:, 1], far[:, 1] - half_b[:, 1])
    overlap_high = torch.minimum(half_a[:, 1], far[:, 1] + half_b[:, 1])
    landing = layout['landing']
    torch.testing.assert_close(landing[:, 0], half_a[:, 0] + gap / 2)
    assert (landing[:, 1] - bridge_half[:, 1] >= overlap_low + BRIDGE['landing_margin'] - eps).all()
    assert (landing[:, 1] + bridge_half[:, 1] <= overlap_high - BRIDGE['landing_margin'] + eps).all()

    start = layout['start']
    torch.testing.assert_close(start[:, 0], landing[:, 0])
    sign = start[:, 1].sign()
    end = torch.maximum(half_a[:, 1], sign * far[:, 1] + half_b[:, 1])
    assert (start[:, 1].abs() - bridge_half[:, 1] >= end + BRIDGE['approach_distance'][0] - eps).all()

    objects, radius = layout['objects'], boxes[..., :2].norm(dim=-1) / 2
    near_edge = (half_a[:, 0] + gap)[:, None]
    band = BRIDGE['object_band']
    assert (objects[..., 0] - radius >= near_edge + band[0] - eps).all()
    assert (objects[..., 0] - radius <= near_edge + band[1] + eps).all()
    assert (objects[..., 0] + radius <= near_edge + 2 * half_b[:, :1] - .8 + eps).all()
    assert ((objects[..., 1] - far[:, 1:]).abs() + radius <= half_b[:, 1:] + eps).all()
    for i, j in ((0, 1), (0, 2), (1, 2)):
        distance = (objects[:, i] - objects[:, j]).norm(dim=-1)
        assert (distance >= radius[:, i] + radius[:, j]).all()
    torch.testing.assert_close(layout['goal'], torch.stack((near_edge[:, 0] + sum(band) / 2, far[:, 1]), -1))

    humans = layout['humans']
    assert (humans.abs() <= half_a[:, None] - .8 + eps).all()
    assert ((humans[:, 0] - humans[:, 1]).norm(dim=-1) >= 1 - eps).all()
    near = layout['near']
    assert .45 < near.float().mean() < .55
    distance = half_a[:, None, 0] - humans[..., 0]
    assert ((distance[near] >= .8 - eps) & (distance[near] <= 2 + eps)).mean(dtype=torch.float) > .99
    depth = BRIDGE['spawn_depth']
    lateral = (humans[..., 1] >= torch.maximum(overlap_low, -half_a[:, 1] + .8)[:, None] - eps) & \
              (humans[..., 1] <= torch.minimum(overlap_high, half_a[:, 1] - .8)[:, None] + eps)
    normal = (distance >= depth[0] - eps) & (distance <= depth[1] + eps) & lateral
    assert normal[~near].mean(dtype=torch.float) > .99


def test_trajectory_and_force():
    start, goal = torch.tensor([[0., -5., .75]]), torch.tensor([[0., 1., .75]])
    duration = torch.tensor([4.])
    for elapsed, expected in ((-1., start), (0., start), (4., goal), (9., goal)):
        pos, vel = bridge_trajectory(start, goal, torch.tensor([elapsed]), duration)
        torch.testing.assert_close(pos, expected)
        torch.testing.assert_close(vel, torch.zeros(1, 3))
    _, vel = bridge_trajectory(start, goal, torch.tensor([2.]), duration)
    assert vel[0, 1] > 0
    state = torch.zeros(1, 13)
    state[0, :3], state[0, 6] = goal, 1
    mass = torch.tensor([1000.])
    torch.testing.assert_close(bridge_control(goal, torch.zeros(1, 3), state, mass, 2., 1., 6.), torch.zeros(1, 3))
    force = bridge_control(goal + torch.tensor([[0., 10., 0.]]), torch.zeros(1, 3), state, mass, 2., 1., 6.)
    torch.testing.assert_close(force, torch.tensor([[0., 6000., 0.]]))


def test_config_uses_1024_samples():
    assert RINGS ** 2 == len(GRID) == 1024


def test_schedule_keeps_arrival_time_with_bounded_speed():
    distance = torch.tensor([6., 6., 6., 0.])
    arrival = torch.tensor([10., 2., 30., 3.])
    duration, wait = bridge_schedule(distance, arrival, BRIDGE['speed_range'], 1 / 30)
    torch.testing.assert_close(wait + duration, arrival)
    torch.testing.assert_close(wait[0], torch.tensor(0.))
    assert wait[1] < 0 and wait[2] > 0
    speed = distance[:3] / duration[:3]
    assert ((speed >= BRIDGE['speed_range'][0] - 1e-6) & (speed <= BRIDGE['speed_range'][1] + 1e-6)).all()
    start, goal = torch.zeros(3, 3), torch.tensor([[0., 6., 0.]]).expand(3, -1)
    pos, _ = bridge_trajectory(start, goal, -wait[:3] + arrival[:3], duration[:3])
    torch.testing.assert_close(pos, goal)


def test_bridge_metadata_rejects_fan_checkpoint():
    task = SimpleNamespace(sensor_metadata={'version': 4, 'bridge': BRIDGE})
    with pytest.raises(ValueError):
        check_sensor_checkpoint({'sensor_metadata': {'version': 3}}, task)
