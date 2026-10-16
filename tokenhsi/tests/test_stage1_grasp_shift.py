import math
import torch

from utils.stage1_grasp_shift import shift_payload_observation, shared_payload_views, initial_grasp_sides


def scene():
    m, o = 2, 4
    node_end = m * 223 + o * 30 + m
    pose_end = node_end + (2 * m + o) * 7
    obs = torch.zeros(1, pose_end + m * 5)
    objects = obs[:, m * 223:m * 223 + o * 30].reshape(1, o, 30)
    objects[..., 6:] = torch.tensor([-.2, -.4, -.2, -.2, -.4, .2,
        -.2, .4, -.2, -.2, .4, .2, .2, -.4, -.2, .2, -.4, .2,
        .2, .4, -.2, .2, .4, .2])
    objects[..., 5] = 2
    poses = obs[:, node_end:pose_end].reshape(1, 8, 7)
    poses[..., 6] = 1
    poses[:, 3, 5:7] = math.sqrt(.5)
    obs[:, pose_end:] = torch.tensor([1, 2, 0, 3, 6, 1, 3, 1, 5, 4])
    return obs, node_end, pose_end


def test_zero_identity_and_no_input_mutation():
    obs, _, _ = scene()
    original = obs.clone()
    assert shift_payload_observation(obs, 2, 4, 0) is obs
    shift_payload_observation(obs, 2, 4, .5)
    assert torch.equal(obs, original)


def test_rotated_center_velocity_and_only_payload_changes():
    obs, node_end, pose_end = scene()
    result = shift_payload_observation(obs, 2, 4, .5)
    expected = obs.clone()
    # Payload slots 1 and 3; slot 1 has a 90-degree Z rotation.
    expected[0, node_end + 3 * 7] = -.2
    expected[0, node_end + 5 * 7 + 1] = .2
    expected[0, 446 + 30] = -.4
    expected[0, 446 + 90] = -.4
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(shift_payload_observation(obs, 2, 4, -.5) - obs,
                               -(result - obs))


def test_packet_order_and_duplicate_payload():
    obs, _, end = scene()
    swapped = obs.clone()
    swapped[:, end:] = obs[:, end:].reshape(1, 2, 5).flip(1).flatten(1)
    a = shift_payload_observation(obs, 2, 4, .5)
    b = shift_payload_observation(swapped, 2, 4, .5)
    torch.testing.assert_close(a[:, :end], b[:, :end])
    obs[0, end + 8] = obs[0, end + 3]
    result = shift_payload_observation(obs, 2, 4, .5)
    torch.testing.assert_close(result[:, :446 + 60], a[:, :446 + 60])


def test_object_permutation():
    obs, start, end = scene()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    permuted = obs.clone()
    permuted[:, 446:566] = obs[:, 446:566].reshape(1, 4, 30)[:, permutation].flatten(1)
    poses = permuted[:, start:end].reshape(1, 8, 7)
    poses[:, 2:6] = obs[:, start:end].reshape(1, 8, 7)[:, 2:6][:, permutation]
    packet = permuted[:, end:].reshape(1, 2, 5)
    packet[..., 3] = inverse[packet[..., 3].long() - 2] + 2
    packet[0, 1, 4] = inverse[packet[0, 1, 4].long() - 2] + 2
    result = shift_payload_observation(permuted, 2, 4, .5)
    original = shift_payload_observation(obs, 2, 4, .5)
    torch.testing.assert_close(result[:, 446:566].reshape(1, 4, 30),
                               original[:, 446:566].reshape(1, 4, 30)[:, permutation])
    torch.testing.assert_close(result[:, start:end].reshape(1, 8, 7)[:, 2:6],
                               original[:, start:end].reshape(1, 8, 7)[:, 2:6][:, permutation])


def test_shared_views_are_opposite_and_keep_same_task_physics():
    obs, start, end = scene()
    obs[:, end:] = torch.tensor([1, 2, 0, 2, 6, 1, 2, 1, 2, 6])
    poses = obs[:, start:end].reshape(1, 8, 7)
    poses[:, 0, 1], poses[:, 1, 1] = -1., 1.
    before = obs.clone()
    signs = initial_grasp_sides(obs)
    torch.testing.assert_close(signs, torch.tensor([[-1., 1.]]))
    views = shared_payload_views(obs, signs, 1.)
    torch.testing.assert_close(views[:, start + 2 * 7 + 1], torch.tensor([-.25, .25]))
    torch.testing.assert_close(views[:, end:], obs[:, end:].expand(2, -1))
    torch.testing.assert_close(views[:, start:start + 14], obs[:, start:start + 14].expand(2, -1))
    torch.testing.assert_close(views[:, start + 6 * 7:end], obs[:, start + 6 * 7:end].expand(2, -1))
    torch.testing.assert_close(obs, before)
    torch.testing.assert_close(shared_payload_views(obs, signs, 0.), obs.expand(2, -1))
    poses[:, 0, 1], poses[:, 1, 1] = 1., -1.
    torch.testing.assert_close(initial_grasp_sides(obs), -signs)


def test_shared_view_batching_and_independent_scene_rejection():
    import pytest
    obs, _, end = scene()
    with pytest.raises(ValueError, match='share'):
        shared_payload_views(obs, torch.tensor([[-1., 1.]]), 1.)
    obs[:, end:] = torch.tensor([1, 2, 0, 2, 6, 1, 2, 1, 2, 6])
    single = shared_payload_views(obs, torch.tensor([[-1., 1.]]), 1.)
    batch = shared_payload_views(obs.expand(2, -1), torch.tensor([[-1., 1.], [1., -1.]]), 1.)
    torch.testing.assert_close(batch, torch.cat((single, single.flip(0))))
