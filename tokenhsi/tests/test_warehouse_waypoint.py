import argparse

import pytest
import torch

from qualitative.warehouse_waypoint import (
    door_box, ontop_waypoint_scene, switch_margin, update_switch, update_return,
    waypoint_scene, worker_spawns,
)
from utils.before_object_alias import alias_layout, goal_aligned_scene
from utils.before_spec import task_objects
from utils.experiments import native_config
from utils.warehouse_coop import sample_roles_graph, sample_layout, box_layout


@pytest.mark.parametrize('value', ['0', '0.5', '1', '2.5', '5.5'])
def test_margin_accepts_half_meter_steps(value):
    assert switch_margin(value) == float(value)


@pytest.mark.parametrize('value', ['-0.5', '1.2', 'nan', 'inf'])
def test_margin_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        switch_margin(value)


def test_root_threshold_and_episode_latch():
    size = torch.tensor([[.6, .8, .4]]).repeat(3, 1)
    roots = torch.tensor([[1.49, 0.], [1.5, 0.], [1.51, 0.]])
    virtual = torch.zeros(3, 2)
    switched, _, radius = update_switch(roots, virtual, size, 1., torch.zeros(3, dtype=torch.bool))
    assert switched.tolist() == [True, True, False]
    torch.testing.assert_close(radius, torch.full((3,), 1.5))
    after, _, _ = update_switch(roots+10, virtual, size, 1., switched)
    assert torch.equal(after, switched)
    reset, _, _ = update_switch(roots+10, virtual, size, 1., torch.zeros_like(switched))
    assert not reset.any()


def test_virtual_box_respects_room_rotation_translation_and_floor():
    origin = torch.tensor([[100., 200., 2.], [0., 0., 0.]])
    shift = torch.tensor([[1., 2.], [0., 0.]])
    result = door_box(origin, shift, torch.tensor([torch.pi/2, 0.]),
        torch.full((2, 2), 4.), .4, .1, torch.tensor([[.5, .5, .4], [.6, .6, .5]]))
    torch.testing.assert_close(result, torch.tensor([[101., 197.8, 2.3], [-4.2, 0., .35]]))


def test_return_waits_for_pickup_uses_payload_and_latches_until_reset():
    size = torch.tensor([[.6, .8, .4]]).repeat(4, 1)
    payload = torch.tensor([[1.49, 0.], [1.5, 0.], [1.51, 0.], [0., 0.]])
    virtual = torch.zeros(4, 2)
    empty = torch.zeros(4, dtype=torch.bool)
    holding = torch.tensor([True, True, True, False])
    picked, returned, _, radius = update_return(holding, payload, virtual, size, 1., empty, empty)
    assert picked.tolist() == [True, True, True, False]
    assert returned.tolist() == [True, True, False, False]
    torch.testing.assert_close(radius, torch.full((4,), 1.5))
    picked, returned, _, _ = update_return(empty, payload+10, virtual, size, 1., picked, returned)
    assert picked.tolist() == [True, True, True, False]
    assert returned.tolist() == [True, True, False, False]
    _, returned, _, _ = update_return(empty, virtual, virtual, size, 1., picked, returned)
    assert returned.tolist() == [True, True, True, False]
    picked, returned, _, _ = update_return(empty, virtual, virtual, size, 1., empty, empty)
    assert not picked.any() and not returned.any()


@pytest.mark.parametrize('family', [0, 1, 3])
@pytest.mark.parametrize('target', [False, True])
def test_only_selected_pose_moves_with_shuffled_packet(family, target):
    packet = torch.tensor([[[1, 2, 2, 6, 14], [1, family, 0, 4 if family == 3 else 0, 6],
                            [1, 4, 1, 5, 13], [1, 4, 3, 7, 15]]], dtype=torch.float)
    packet = packet[:, [2, 0, 3, 1]]
    poses = torch.randn(1, 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    source = torch.cat((torch.randn(1, 1136), poses.flatten(1), packet.flatten(1)), -1)
    saved = source.clone()
    expected = goal_aligned_scene(source, 4, 8)
    modified = waypoint_scene(source, torch.tensor([2]), torch.tensor([[8., 9., 10.]]),
                              target=torch.tensor([target]))
    got = modified[:, 1136:1248].reshape(1, 16, 7)
    _, _, alias, _ = alias_layout(packet, 4, 8)
    torch.testing.assert_close(got[0, alias[0, 3], :3], poses[0, 14, :3])
    moved, kept = (14, 6) if target else (6, 14)
    torch.testing.assert_close(got[0, kept], poses[0, kept])
    torch.testing.assert_close(got[0, moved, :3], torch.tensor([8., 9., 10.]))
    got[0, moved, :3] = poses[0, moved, :3]
    torch.testing.assert_close(modified, expected)
    assert torch.equal(source, saved)


@pytest.mark.parametrize('target', [False, True])
def test_ontop_waypoint_moves_only_the_selected_dependent_ontop_pose_in_gta(target):
    # Same three follower kinds and one solo ON_TOP, with shuffled task rows.
    packet = torch.tensor([[1, 3, 1, 5, 4], [1, 4, 3, 7, 15],
                           [1, 2, 0, 4, 12], [1, 4, 2, 6, 14]]).float().repeat(4, 1, 1)
    packet[1:3, 0, 1] = torch.tensor([0., 1.])
    packet[1:3, 0, 3] = 0
    packet[3, 0, 4] = 13
    poses = torch.randn(4, 16, 7)
    scene = torch.cat((torch.randn(4, 1136), poses.flatten(1), packet.flatten(1)), -1)
    snapshot = scene.clone()
    expected = goal_aligned_scene(scene, 4, 8)
    entrance = torch.tensor([[8., 9., 10.]]).expand(4, -1)
    modified = ontop_waypoint_scene(scene, torch.ones(4, dtype=torch.long), entrance,
                                    target=torch.full((4,), target))
    _, _, alias, _ = alias_layout(packet, 4, 8)
    slot = int(alias[0, 0]) if target else 5
    expected[:, 1136:1248].reshape(4, 16, 7)[0, slot, :3] = entrance[0]
    torch.testing.assert_close(modified, expected, rtol=0, atol=0)
    assert torch.equal(scene, snapshot)


def test_worker_spawns_keep_pushers_clearances_and_near_routes():
    torch.manual_seed(42)
    env = native_config('stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress')['env']
    cfg = env['sensorScene']['warehouse']
    n = 128
    sizes = torch.tensor([.5, .5, .4]).expand(n, 4, 3)
    counts, direction, mode = torch.full((n,), 2), torch.full((n,), 2), torch.zeros(n, dtype=torch.long)
    _, pre, tasks, roles = sample_roles_graph(sizes, env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'], counts, 1.)
    near = (pre >= 0) & (torch.rand(n, 4) < .5)
    layout = sample_layout(tasks, pre, roles, sizes, sizes, near, counts, direction, mode, cfg, 16)
    bank = box_layout(counts, direction, mode, cfg)
    roots = layout['root']
    objects = torch.cat((layout['source'], layout['support'], bank['source'][:, None]), 1)
    dimensions = torch.cat((sizes[..., :2], sizes[..., :2], bank['world_size'][:, None]), 1)
    valid = task_objects(pre, tasks)
    valid[:, :4] &= tasks != 5
    valid = torch.cat((valid, torch.ones(n, 1, dtype=torch.bool)), 1)
    rows = torch.arange(n)
    follower = (pre >= 0).long().argmax(-1)
    leader = pre[rows, follower]
    goal = layout['goal'][rows, leader]
    got, good = worker_spawns(roots, objects, dimensions, valid, leader, follower,
        layout['source'], goal, tasks[rows, follower], near[rows, follower])
    good &= layout['valid']
    assert good.sum() >= 32
    push = tasks == 5
    torch.testing.assert_close(got[push], roots[push], rtol=0, atol=0)
    p = got[rows, leader]
    assert ((p[good, 0] >= -3.5) & (p[good, 0] <= -2.5)).all()
    assert ((p[good, 1].abs() >= 2.4) & (p[good, 1].abs() <= 3.2)).all()
    assert (p[good, 1]*layout['source'][rows, leader, 1][good] >= 0).all()
    distance = torch.cdist(got, got) + torch.eye(4)[None]*100
    assert (distance[good] >= .85).all()
    for person in (leader, follower):
        gap = ((got[rows, person, None]-objects).abs()-dimensions/2).clamp_min(0).norm(dim=-1)
        assert ((gap >= .5) | ~valid)[good].all()
    first = (p-layout['source'][rows, leader]).norm(dim=-1)+(layout['source'][rows, leader]-goal).norm(dim=-1)
    target = torch.where((tasks[rows, follower] < 3)[:, None], goal, layout['source'][rows, follower])
    route = (got[rows, follower]-target).norm(dim=-1)
    route += torch.where(tasks[rows, follower] == 4, (layout['source'][rows, follower]-goal).norm(dim=-1), 0.)
    assert (route < first-.001)[good & near[rows, follower]].all()
