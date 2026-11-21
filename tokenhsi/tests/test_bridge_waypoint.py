import argparse

import pytest
import torch

from qualitative.bridge_waypoint import bridge_goal, switch_margin, update_switch, waypoint_scene
from utils.before_object_alias import alias_layout, goal_aligned_scene


@pytest.mark.parametrize('value', ['0', '0.5', '1', '2.5', '5', '5.5'])
def test_margin_accepts_half_meter_steps(value):
    assert switch_margin(value) == float(value)


@pytest.mark.parametrize('value', ['-0.5', '1.2', 'nan', 'inf'])
def test_margin_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        switch_margin(value)


def test_payload_threshold_includes_box_radius_and_latches():
    size = torch.tensor([[.6, .8, .4]]).repeat(3, 1)
    xy = torch.tensor([[1.49, 0.], [1.5, 0.], [1.51, 0.]])
    goal = torch.zeros(3, 2)
    switched, _, radius = update_switch(xy, goal, size, 1., torch.zeros(3, dtype=torch.bool))
    assert switched.tolist() == [True, True, False]
    torch.testing.assert_close(radius, torch.full((3,), 1.5))
    after, _, _ = update_switch(xy+10, goal, size, 1., switched)
    assert torch.equal(after, switched)


def test_goal_is_payload_center_above_bridge_top():
    bridge = torch.tensor([[2., 3., .75, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
    goal = bridge_goal(bridge, torch.tensor([[3.2, 2., 1.5]]), torch.tensor([[.5, .5, .4]]))
    torch.testing.assert_close(goal, torch.tensor([[2., 3., 1.7]]))


@pytest.mark.parametrize('family', [0, 1, 3])
def test_only_leader_target_moves_and_follower_alias_stays_at_original_goal(family):
    packet = torch.tensor([[[1, 2, 0, 4, 12], [1, family, 1, 5 if family == 3 else 0, 4],
                            [1, 0, 2, 0, 6], [1, 0, 3, 0, 7]]], dtype=torch.float)
    poses = torch.randn(1, 16, 7)
    poses[..., 3:7] = torch.tensor([0., 0., 0., 1.])
    source = torch.cat((torch.zeros(1, 1136), poses.flatten(1), packet.flatten(1)), -1)
    saved = source.clone()
    expected = goal_aligned_scene(source, 4, 8)
    modified = waypoint_scene(source, torch.tensor([0]), torch.tensor([[8., 9., 10.]]))
    got = modified[:, 1136:1248].reshape(1, 16, 7)
    _, _, alias, _ = alias_layout(packet, 4, 8)
    torch.testing.assert_close(got[0, alias[0, 1], :3], poses[0, 12, :3])
    torch.testing.assert_close(got[0, 12, :3], torch.tensor([8., 9., 10.]))
    got[0, 12, :3] = poses[0, 12, :3]
    torch.testing.assert_close(modified, expected)
    assert torch.equal(source, saved)
